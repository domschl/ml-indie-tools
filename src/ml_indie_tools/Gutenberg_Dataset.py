import logging
import os
import re
import time

from enum import Enum
from urllib.request import urlopen

class Gutenberg_Dataset():
    """ A fuzzy, lightweight class to access, search and filter Project Gutenberg resources
    
    GutenbergLib by default uses a mirror's root URL. Alternatively, you can specify a local directory containing a Gutenberg mirror.
    That mirror directory needs to contain a GUTINDEX.ALL file and has typically many 
    sub-directories `0` ,.. `n` .

    A mirror of project Gutenberg can be created by:
    
    .. code-block:: console

        #!/bin/bash
        rsync -zarv --dry-run --prune-empty-dirs --del --include="*/" --include='*.'{txt,pdf,ALL} --exclude="*" aleph.gutenberg.org::gutenberg ./gutenberg_mirror

    You can remove the PDF files, since they are currently not used, and need to review the `--dry-run` option.

    Note: :func:`~Gutenberg_Dataset.Gutenberg_Dataset.load_index` needs to be called before any other methods.
    
    :param root_url: url of Project Gutenberg or any mirror URL, or a local directory containing a Gutenberg mirror.
    :param cache_dir: path to a directory that will be used to cache the Gutenberg index and already downloaded texts. The cache directory is only used, if a remote Gutenberg URL and not a local mirror is used.
    """
    def __init__(self, root_url="https://www.gutenberg.org/dirs", cache_dir="gutenberg"):
        # old root, vanished: http://www.mirrorservice.org/sites/ftp.ibiblio.org/pub/docs/books/gutenberg
        self.log = logging.getLogger('GutenbergLib')
        self.root_url = root_url
        self.index=None
        self.NEAR=2048
        self.start_tokens=["*** START OF THIS PROJECT", "E-text prepared by", 
                           "This book was generously provided by the ", "*** START OF THIS PROJECT GUTENBERG",
                           "START OF THE PROJECT GUTENBERG"]    
        self.near_start_tokens=["produced by ", "Produced by ", ", prepared by", "Transcriber's Note", 
                                "Transcriber's note:", "Anmerkungen zur Tanskription", 
                                "Distributed Proofreading Team", "offensichtliche Schreibfehler",
                                "Inkonsistenzen in der Rechtschreibung", "Im Original",
                                "Obvious printer errors", "spelling was kept",
                                "ERRATA have been applied", "punctuation errors", "have been silently corrected",
                                "changes to the text", "Transcriber note", "Transcriber Note","_italic_",
                                "Variable spelling"] 
        self.end_tokens=["End of the Project Gutenberg", "*** END OF THIS PROJECT", 
                         "***END OF THE PROJECT GUTENBER", "Ende dieses Projekt Gutenberg", 
                         "*** END OF THE PROJECT GUTENBERG",
                         "End of Project Gutenberg", "Transcriber's Note"]
        self.local_mirror=False
        if root_url[:4]!='http':
            if not os.path.exists(root_url):
                self.log.error(f'If root_url points to non-http URL, it must be an existing local directory containing a Gutenberg mirror: {root_url}')
            else:
                index_path=os.path.join(root_url, "GUTINDEX.ALL")
                if not os.path.exists(index_path):
                    self.log.error(f'GUTINDEX.ALL not found in {root_url}, this is not a valid Gutenberg mirror')
                else:
                    self.local_mirror=True
                    self.cache_dir=None
            return
        if self.local_mirror is False:
            try:
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                self.cache_dir=cache_dir
            except Exception as e:
                self.cache_dir=None
                self.log.error(f"Failed to create cache directory {cache_dir}, {e}")

    def _parse_record(self, record, verbose=True):
        """ internal function to recreate some consistent record information from near-freestyle text """
        rl=record.split('\n')
        white=str(chr(160))+str(chr(9))+" " # non-breaking space, TAB, and space
        ebook_no=""
        while len(rl[0])>0 and rl[0][-1] in white:
            rl[0]=rl[0][:-1]
        while len(rl[0])>0 and not rl[0][-1] in white:
            ebook_no=rl[0][-1]+ebook_no
            rl[0]=rl[0][:-1]
        while len(rl[0])>0 and rl[0][-1] in white:
            rl[0]=rl[0][:-1]
        
        # Sanity check
        try:
            fa=re.findall(ebook_no,"\A[0-9]+[A-C]\Z")
        except Exception as e:
            fa=None
            if verbose is True:
                self.log.warning(f"Failed to apply regex on >{ebook_no}<: {e}")
            
        if len(rl[0])<5 or fa==None or len(ebook_no)>7:
            if verbose is True:
                print("-------------------------------------")
                print(record)
                print("- - - - - - - - - - - - - - - - - - -")
                print(f"Dodgy record: {rl[0]}")
                print(f"    ebook-id:  >{ebook_no}<")
            return None
        
        for i in range(len(rl)):
            rl[i]=rl[i].strip()
            
        p=0
        while p<len(rl)-1:
            if len(rl[p+1])==0:
                print(f"Invalid rec: {record}")
                p+=1
            else:
                if rl[p+1][0]!="[":
                    rl[p]+=" "+rl[p+1]
                    del rl[p+1]
                    if rl[p][-1]==']':
                        p+=1
                else:
                    p+=1
        
        rec={}
        l0=rl[0].split(", by ")
        rec['title']=l0[0]
        rec['ebook_id']=ebook_no
        # if len(l0)>2:
        #    print(f"Chaos title: {rl[0]}")
        if len(l0)>1:
            rec['author']=l0[-1]
        for r in rl[1:]:
            if r[0]!='[' or r[-1]!=']':
                if r[0]=='[':
                    ind=r.rfind(']')
                    if ind != -1:
                        # print(f"Garbage trail {r}")
                        r=r[:ind+1]
                        # print(f"Fixed: {r}")
                    else:
                        # print(f"Missing closing ] {r}")
                        r+=']'
                        # print(f"Fixed: {r}")
            if r[0]=='[' and r[-1]==']':
                r=r[1:-1]
                i1=r.find(':')
                if i1==-1:
                    r=r.replace("Author a.k.a.","Author a.k.a.:")
                    i1=r.find(':')
                if i1!=-1:
                    i2=r[i1:].find(' ')+i1
                else:
                    i2=-1
                if i1==-1 and i2==-1:
                    pass
                    # print(f"Invalid attribut in {rl}::{r}")
                else:
                    if i2-i1==1:
                        key=r[:i1]
                        val=r[i2+1:]
                        if '[' in key or ']' in key or '[' in val or ']' in val or len(key)>15:
                            pass
                            # print("messy key/val")
                        else:
                            rec[key.strip().lower()]=val.strip()
                    else:
                        pass
                        # print(f"Bad attribute name terminator, missing ': ' {r}")
            else:
                pass
                # print(f"Invalid attribut in {rl}::{r}")
        if len(rec)>1:
            if "language" not in rec.keys():
                rec["language"]="English"
        return rec
        
    def _parse_index(self, lines):
        """ internal function to parse the fuzzy text-based Gutenberg table of content """
        class State(Enum):
            NONE=1,
            SYNC_START=2,
            SYNC_REC=3,
            END=5
    
        white=str(chr(160))+str(chr(9))+" " # non-breaking space, TAB, and space
        state=State.NONE
        start_token="~ ~ ~ ~"
        stop_token=["====="]
        end_token="<==End"
        ignore_headers=["TITLE and AUTHOR"]
        ignore_content=["Not in the Posted Archives","human-read audio ebooks", "Audio:"]
        empty_lines=0
        records=[]
        for line in lines:
            if line[:len(end_token)]==end_token:
                state=State.END
                break

            if state==State.NONE:
                if line[:len(start_token)]==start_token:
                    state=State.SYNC_START
                    empty_lines=0
                    continue
            if state==State.SYNC_START:
                if len(line.strip())==0:
                    empty_lines+=1
                    if empty_lines>1:
                        state=State.NONE
                        continue
                else:
                    stopped=False
                    for stop in stop_token:
                        if line[:len(stop)]==stop:
                            stopped=True
                            break
                    if stopped is True:
                        state=State.NONE
                        empty_lines=0
                        continue
                    ignore=False
                    for header in ignore_headers:
                        if line[:len(header)]==header:
                            empty_lines=0
                            ignore=True
                    for token in ignore_content:
                        if token in line:
                            empty_lines=0
                            ignore=True
                    if ignore is True:
                        continue
                    rec=line
                    state=State.SYNC_REC
                    continue
            if state==State.SYNC_REC:
                if len(line.strip())==0 or line[0] not in white:
                    if len(records)<10:
                        parsed_rec=self._parse_record(rec, verbose=True)
                    else:
                        parsed_rec=self._parse_record(rec, verbose=False)
                        
                    if parsed_rec is not None:
                        records.append(parsed_rec)
                    empty_lines=1
                    if len(line.strip())==0:
                        state=State.SYNC_START
                        continue
                    else:
                        rec=line
                        continue
                rec=rec+"\n"+line
        return records
                    
    def load_index(self, cache=True, cache_expire_days=30):
        """ This function loads the Gutenberg record index, either from cache, or from a website
        
        This should be the first method being used, since many other methods rely on the index being loaded.

        :param cache: default `True`, use the cache directory to cache both index and text files. Index expires after `cache_expire_days`, text files never expire. Should *NOT* be set to `False` in order to prevent unnecessary re-downloading.
        :param cache_expire_days: Number of days after which the index is re-downloaded.
        """
        raw_index=None
        if self.local_mirror is False:
            if self.cache_dir is None:
                self.log.error("Cannot cache library index, no valid cache directory.")
                return False
            ts_file=os.path.join(self.cache_dir,"timestamp")
            cache_file=os.path.join(self.cache_dir,"gutenberg_index")
            expired=True
            read_from_cache=False
            if os.path.isfile(ts_file) and os.path.isfile(cache_file):
                try:
                    with open(ts_file,'r') as f:
                        ts=float(f.read())
                    if time.time()-ts<cache_expire_days*24*3600:
                        expired=False
                        read_from_cache = True
                        self.log.debug("Cache timestamp read.")
                    else:
                        self.log.debug("Cache for Gutenberg-index is expired, reloading from web.")
                except:
                    self.log.warning("Failed to read cache timestamp, reloading from web.")
            if expired is False and os.path.isfile(cache_file):
                try:
                    with open(cache_file,'r') as f:
                        raw_index=f.read()
                        self.log.debug(f"Gutenberg index read from local cache: {cache_file}")
                except:
                    expired=True
                    self.log.warning("Failed to read cached index, reloading from web.")
            if expired is True:
                index_url=self.root_url+"/GUTINDEX.ALL"
                try:
                    raw_index = urlopen(index_url).read().decode('utf-8')
                    if raw_index[0]=='\ufeff':  # Ignore BOM
                        raw_index=raw_index[1:]
                    raw_index=raw_index.replace('\r','')
                    self.log.debug(f"Gutenberg index read from {index_url}")
                except Exception as e:
                    self.log.error(f"Failed to download Gutenberg index from {index_url}, {e}")
                    return False
            if cache is True and read_from_cache is False:
                try:
                    with open(ts_file,'w') as f:
                        f.write(str(time.time()))
                        self.log.debug("Wrote read cache timestamp.")
                except Exception as e:
                    self.log.error(f"Failed to write cache timestamp to {ts_file}, {e}")
                try:
                    with open(cache_file,'w') as f:
                        f.write(raw_index)
                        self.log.debug("Wrote read cached index.")
                except Exception as e:
                    self.log.error(f"Failed to write cached index to {cache_file}, {e}")
        else:
            index_file=os.path.join(self.root_url, "GUTINDEX.ALL")
            try:
                with open(index_file,'r') as f:
                    raw_index=f.read()
                    if raw_index[0]=='\ufeff':  # Ignore BOM
                        raw_index=raw_index[1:]
                    raw_index=raw_index.replace('\r','')
                    self.log.debug(f"Gutenberg index read from local mirror: {index_file}")
            except Exception as e:
                self.log.error(f"Failed to read Gutenberg index from local mirror: {index_file}, {e}")
                return
        lines=raw_index.split('\n')
        self.records=self._parse_index(lines)

    def load_book(self, ebook_id):
        """ get text of an ebook from Gutenberg by ebook_id 
        
        This function returns the unfiltered raw text including all Gutenberg headers and footers.
        Use :func:`~Gutenberg_Dataset.Gutenberg_Dataset.get_book` to retrieve a dictionary with metadata and filtered text.

        :param ebook_id: Gutenberg id (Note: string, since this sometimes contains a character!)
        :returns: book text as string, unfiltered. Can be filtered with :func:`~Gutenberg_Dataset.Gutenberg_Dataset.filter_text`
        """
        txt, dl, val = self._load_book_ex(ebook_id)
        if val is True:
            return txt, dl
        else:
            return None, dl

    def _read_download(self, filenames, path_stub, cache_name):
        """ Internal function to read ebook from cache or download it. """
        cache_file=None
        downloaded=False
        if self.cache_dir is not None:
            cache_file=os.path.join(self.cache_dir,cache_name)
            if os.path.isfile(cache_file):
                try:
                    with open(cache_file,'r') as f:
                        data=f.read()
                        self.log.debug(f"Book read from cache at {cache_file}")
                        downloaded = False
                        return data, None, downloaded
                except Exception as e:
                    self.log.error(f"Failed to read cached file {cache_file}")
        data=None
        file_url=None
        for filename, encoding in filenames:
            file_url=self.root_url+path_stub+filename
            if self.local_mirror is False:
                try:
                    if encoding != 'bin':
                        data = urlopen(file_url).read().decode(encoding)
                    else:
                        data = urlopen(file_url).read()
                    self.log.debug(f"Book read from {file_url}")
                    downloaded = True
                    break
                except Exception as e:
                    self.log.debug(f"Failed to download {file_url}, {e}")
            else:
                try:
                    if encoding != 'bin':
                        with open(file_url,'r',encoding=encoding) as f:
                            data = f.read()
                    else:
                        with open(file_url,'rb') as f:
                            data = f.read()
                    self.log.debug(f"Book read from local mirror at {file_url}")
                    downloaded = False
                    break
                except Exception as e:
                    self.log.debug(f"Failed to read local mirror at {file_url}, {e}")
        return data, cache_file, downloaded

    def _load_book_ex(self, ebook_id):
        """ Internal function to get text of an ebook from Gutenberg by ebook_id with remote download information
        
        This function returns the unfiltered raw text including all Gutenberg headers and footers and a boolean flag
        indicating with 'True', if the book was downloaded from a remote source, 'False' indicates cached book retrieval
        without remote access.

        The flag can be used to control or limit the number of books downloaded from the remote source.

        Use :func:`~Gutenberg_Dataset.Gutenberg_Dataset.get_book` to retrieve a dictionary with metadata and filtered text.

        :param ebook_id: Gutenberg id (Note: string, since this sometimes contains a character!)
        :returns: tuple: book text as string, unfiltered, and a flag indicating with 'True' if book was downloaded from remote, validity flag, 'True' indicates valid text. 
        """
        if ebook_id is None or len(ebook_id)==0:
            self.log.error("No ebook_id given.")
            return None, None, None
        if ebook_id[-1]=='C':
            ebook_id=ebook_id[:-1]
        path_stub=""
        downloaded = False
        valid = False
        
        for i in range(len(ebook_id)-1):
            path_stub+="/"+ebook_id[i]
        path_stub+="/"+ebook_id+"/"
        filenames=[(ebook_id+"-0.txt",'utf-8'), (ebook_id+".txt",'utf-8'), (ebook_id+"-8.txt","latin1"), (ebook_id+".txt","latin1")]
        cache_name=ebook_id+".txt"
        data, cache_file, downloaded = self._read_download(filenames, path_stub, cache_name)
        if data is not None:
            if data[0]=='\ufeff':  # Ignore BOM
                data=data[1:]
            data=data.replace('\r','')
            valid = True
        else:
            filenames = [(ebook_id+"-pdf.pdf","bin")]
            cache_name=ebook_id+".pdf"
            data, cache_file, downloaded = self._read_download(filenames, path_stub, cache_name)
            if data is not None:
                self.log.error(f"Ebook {cache_name} is only available in PDF format, this is not supported.")
            else:
                self.log.warning(f"Failed to download {filenames}, skipping book.")
                return None, downloaded, False
        if cache_file is not None:
            try:
                with open(cache_file,'w') as f:
                    f.write(data)
            except:
                self.log.error(f"Failed to cache file {cache_file}")
        return data, downloaded, valid
    
    def filter_text(self, book_text, add_start_tokens=None, add_near_start_tokens=None, add_end_tokens=None):
        """ Heuristically remove header and trailer texts not part of the actual books

        Unfortunatelly, formatting of Gutenberg books is an unbelievable mess. Using lists of tokens `self.start_tokens` (indicating
        the start of the actual book text), `self.near_start_tokens` (indicating possibly ambiguous tokens near a `start_tokens` token, 
        further narrowing the start of text), and `self.end_tokens` (indicating the end of the book text), this function tries to find 
        the start and end of the book text. The user can either extend the lists of class member tokens, of provide temporary additional 
        tokens as parameter to this function.
        
        The list of `start_tokens` contains only tokens that are always significant as being part of header-cruft (e.g. 'START OF THIS GUTENBERG'). 
        `near_start_tokens` are tokens that might be ambiguous, but are still part of the header-cruft, (e.g. 'produced by'). 
        `near_start_tokens` are only used, if they are within `self.NEAR` bytes to the latest `start_tokens` token, 
        to heuristically prevent false positives.

        *Note:* Use logging via `logging.basicConfig(level=logging.DEBUG)` to analyze the filtering process.

        :param book_text: text of the book (string)
        :param add_start_tokens: additional start tokens (list of strings)
        :param add_near_start_tokens: additional near start tokens (list of strings)
        :param add_end_tokens: additional end tokens (list of strings)
        :returns: filtered text (string)
        """
        start_tokens = self.start_tokens
        if add_start_tokens is not None:
            start_tokens.extend(add_start_tokens)
        near_start_tokens = self.near_start_tokens
        if add_near_start_tokens is not None:
            near_start_tokens.extend(add_near_start_tokens)
        end_tokens = self.end_tokens
        if add_end_tokens is not None:
            end_tokens.extend(add_end_tokens)

        if book_text is None:
            self.log.warning("Filter: book text is None, returning None")
            return None
        blen=len(book_text)
        
        pstart=0
        for token in start_tokens:
            pos=book_text.find(token)
            if pos > pstart:
                pstart = pos
                self.log.debug(f"Start-token [{token}] found at position {pos}")
        if pstart>0:
            pos=book_text[pstart:].find("\n\n")
            if pos>=0 and pos <= self.NEAR:
                pos += pstart
                while book_text[pos]=='\n':
                    pos += 1  # eof?!
                pstart=pos
        if pstart>blen/2:
            self.log.warning("Preamble is taking more than half of the book!")
        new_book=book_text[pstart:]
        xpos=-1
        for token in near_start_tokens:
            pos=new_book.find(token)
            if pos>=0 and pos<=self.NEAR:
                self.log.debug(f"Near-Start-token [{token}] found at position {pos}")
                if pos>xpos:
                    xpos=pos
        if xpos > -1:
            pos2=new_book[xpos:].find("\n\n")
            if pos2<=self.NEAR and pos2>0:
                self.log.debug(f"Trying extra skipping (2) for {pos2}...")
                while new_book[xpos+pos2]=='\n':
                    pos2 += 1
                new_book=new_book[xpos+pos2:]
                self.log.debug(f"Additionally shortened start by {xpos+pos2} chars")
            else:
                pos2=new_book[xpos:].find("\n")
                if pos2<=self.NEAR and pos2>0:
                    self.log.debug(f"Trying extra skipping (3) for {pos2}...")
                    while new_book[xpos+pos2]=='\n':
                        pos2 += 1
                    new_book=new_book[xpos+pos2:]
                    self.log.debug(f"Additionally shortened start by {xpos+pos2}, {xpos}+{pos2} chars")
                else:
                    pos2=0
                    new_book=new_book[xpos+pos2:]
        
        pend=len(new_book)
        for token in end_tokens:
            pos=new_book.find(token)
            if pos!=-1 and pos < pend:
                self.log.debug(f"End-token [{token}] found at pos {pos}")
                pend = pos
        if pend<len(new_book):
            pos=new_book[:pend].rfind("\n\n")
            if pos>0:
                while new_book[pos]=='\n':
                    pos -= 1  # eof?!
                pend=pos+1
        else:
            self.log.debug("No end token found!")
        if pend<len(new_book)/2:
            self.log.debug("End-text is taking more than half of the book!")
        new_book=new_book[:pend]
        return new_book
        
    def find_keywords(self,*search_keys):
        """ Search of an arbitrary number of keywords in a book record
       
        *Note:* :func:`~Gutenberg_Dataset.Gutenberg_Dataset.load_index` needs to be called once before this function can be used.

        :returns: list of records that contain all keywords in any field.
        """
        frecs=[]
        for rec in self.records:
            found=True
            for sk in search_keys:
                subkey=False
                for key in rec.keys():
                    if sk.lower() in key.lower() or sk.lower() in rec[key].lower():
                        subkey=True
                        break
                if subkey is False:
                    found=False
                    break
            if found is True:
                frecs += [rec]
        return frecs
    
    def search(self, search_dict):
        """ Search for book record with key specific key values
        For a list of valid keys, use `get_record_keys()`
        Standard keys are: `ebook_id`, `author`, `language`, `title`

        *Note:* :func:`~Gutenberg_Dataset.Gutenberg_Dataset.load_index` needs to be called once before this function can be used.

        Example: `search({"title": ["philosoph","phenomen","physic","hermeneu","logic"], "language":"english"})`
        Find all books whose titles contain at least one of the keywords, language english. Search keys can either be
        search for a single keyword (e.g. english), or an array of keywords. 
        
        :returns: list of records """
        if not hasattr(self, 'records') or self.records is None:
            self.log.debug("Index not loaded, trying to load...")
            self.load_index()
        frecs=[]
        for rec in self.records:
            found=True
            for sk in search_dict:
                if sk not in rec:
                    found=False
                    break
                else:
                    skl=search_dict[sk]
                    if not isinstance(skl,list):
                        skl=[skl]
                    nf=0
                    for skli in skl:
                        if skli.lower() in rec[sk].lower():
                            nf=nf+1
                    if nf==0:
                        found=False
                        break
            if found is True:
                frecs += [rec]
        return frecs
        
    def insert_book_texts(self, search_dict, download_count_limit=20):
        """ Inserts book texts into the records returned by :func:`~Gutenberg_Dataset.Gutenberg_Dataset.search`.

        In order to prevent the download of too many books, the download count limit is set to `download_count_limit`.
        Downloaded books are cached and cached books are not counted towards the download count limit. Calling this
        function again will download books that have not been downloaded yet. The filtered book content is inserted
        into the dictionary with the key `text`.
        
        :param search_dict: search array of dictionaries that at least contain the key `ebook_id`.
        :param download_count_limit: maximum number of books to download, if no local mirror is used. No limits apply for local mirrors.
        :returns: list of records including filtered book text-based in the `text` field.
        """
        dls=0
        for i in range(0, len(search_dict)):
            self.log.debug(f"Getting id={search_dict[i]['ebook_id']}, {search_dict[i]['title']}")
            bt, dl, val = self._load_book_ex(search_dict[i]["ebook_id"])
            if bt is None or val is False:
                if val is False:
                    self.log.warning(f"Download of book {search_dict[i]['ebook_id']}, {search_dict[i]['title']}: invalid format!")
                else:
                    self.log.error(f"Download of book {search_dict[i]['ebook_id']}, {search_dict[i]['title']} failed!")
                continue
            search_dict[i]['text']=self.filter_text(bt)
            if dl is True and self.local_mirror is False:
                dls += 1
                if dls>download_count_limit:
                    self.log.error(f"Download limit reached ({download_count_limit}), stopping download...")
                    break
        return search_dict  

    def get_book(self, ebook_id: str):
        """ Get a book record metadata and filtered text by its ebook_id
       
        This function returns a dictionary with metadata and filtered text. Use :func:`~Gutenberg_Dataset.Gutenberg_Dataset.load_book` 
        to get the raw unfiltered text.
        
        *Note:* :func:`~Gutenberg_Dataset.Gutenberg_Dataset.load_index` needs to be called once before this function can be used.

        :param ebook_id: ebook_id (String, since some IDs contain letters) of the book to be retrieved
        :returns: book record (dictionary with metadata and filtered text)
        """
        for rec in self.records:
            if rec["ebook_id"]==ebook_id:
                text, _, valid = self._load_book_ex(ebook_id)
                if text is None or valid is False:
                    self.log.Error(f"Download of book {ebook_id} failed!")
                    return None
                rec['text']=self.filter_text(text)
                return rec
        return None

    def get_record_keys(self):
        """ Get a list of all keys that are used within records. 
        Standard keys are: `ebook_id`, `author`, `language`, `title`.
        
        *Note:* :func:`~Gutenberg_Dataset.Gutenberg_Dataset.load_index` needs to be called once before this function can be used.

        :returns: list of all different keys that are somehow used."""
        rks=[]
        for r in self.records:
            rks=set(list(rks) + list(r.keys()))
        return rks

    def get_unique_record_values(self, key):
        """ Get a list of all unique values a given keys has for all records.
        
        *Note:* :func:`~Gutenberg_Dataset.Gutenberg_Dataset.load_index` needs to be called once before this function can be used.

        Example: `get_unique_records_values('language')` returns all languages in Gutenberg.

        :param key: key to search for.
        :returns: list of all unique values for a given key.
        """
        uv=[]
        if key not in self.get_record_keys():
            print(f"{key} is not a key used in any record!")
            return None
        for r in self.records:
            if key in r:
                uv=set(list(uv)+[r[key]])
        uv=sorted(uv)
        return uv
