import random
import logging
from itertools import Counter
try:
   from IPython.core.display import display, HTML
except:
    pass

class Text_Dataset:
    """ Initialize the Text_Dataset with a list of texts.

    The Gutenberg_Dataset can be used to create such a list, by:
        
    .. code-block:: python

        from ml_indie_tools.Gutenberg_Dataset import Gutenberg_Dataset
        from ml_indie_tools.Text_Dataset import Text_Dataset
        gd = Gutenberg_Dataset()
        gd.load_index()
        ls = gd.search({'author': 'kant', 'title': 'kritik', 'language': 'german'})  # returns a list of texts
        ls = gd.insert_texts(ls)  # this inserts the actual text of the books into field 'text'.
        # Now ls contains a valid list of text records:
        td = Text_Dataset(ls)
    
    :param text_list: list of text-records of the form: {'author': 'author', 'title': 'title', 'language': 'some-language', 'text': 'the-long-text'}. Optinal parameters: 'weight': 1.0
    :param sanitize_white_space: If True, white space is replaced by a single space.
    :param separate_punctuation: If True, punctuation is separated from words.
    :param preserve_case: If True, the case of the text is preserved.
    """
    def __init__(self, text_list, sanitize_white_space=False, separate_punctuation=False, preserve_case=True):
        self.log = logging.getLogger("Datasets")
        self.text_list = []
        self.index = 1
        self.word_tokenizer_init = False
        self.char_tokenizer_init = False
        self.getitem_init = False
        self.tokenizer_type = None
        
        req_attrs=['title', 'author', 'language', 'text']
        for ind in range(0,len(text_list)):
            valid=True
            miss=[]
            for attr in req_attrs:
                if attr not in text_list[ind]:
                    valid=False
                    miss.append(attr)
            if valid is False:
                self.log.error(f"Missing attribute(s) {miss} in text[{ind}], skipping")
                continue
            text=text_list[ind]
            text['index']=self.index
            text['text']=self.filter_text(text['text'], sanitize_white_space=sanitize_white_space, separate_punctuation=separate_punctuation, preserve_case=preserve_case)
            self.index += 1
            self.text_list.append(text)
        lt=len(self.text_list)
        if lt==1:
            self.log.info(f"Loaded {lt} text")
        else:
            self.log.info(f"Loaded {lt} texts")
        self._calc_probability_weights()

    def _calc_probability_weights(self):
        prs = 0
        for text in self.text_list:
            if 'weight' in text:
                w=text['weight']
            else:
                w=1.0
            pr = len(text['text']) * w
            prs = prs + pr
            text['probability_weight'] = pr
        for text in self.text_list:
            text['probability_weight'] = text['probability_weight'] / prs
        self.tidx = []
        self.tcum = []
        tc=0
        for idx in range(0,len(self.text_list)):
            self.tidx.append(idx)
            text = self.text_list[idx]
            self.tcum.append(text['probability_weight'] + tc)
            tc = self.tcum[-1]

    def _get_random_text_index(self, weighted=True):
        """ Return a random text index from the Text_Dataset.
        
        :param weighted: If True, the probability of a text is weighted by its calculated 'probability_weight' attribute.
        :return: a random text index
        """
        if weighted is True:
            return random.choices(self.tidx, self.tcum)[0]
        else:
            return random.choice(self.tidx)

    def filter_text(self, text, sanitize_white_space=False, separate_punctuation=False, preserve_case=True):
        """ Filter a text.
        
        :param text: text to filter
        :param sanitize_white_space: If True, white space is replaced by a single space.
        :param separate_punctuation: If True, punctuation is separated from words.
        :param preserve_case: If True, the case of the text is preserved.
        :return: filtered text
        """
        if preserve_case is False:
            text = text.lower()
        punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        if separate_punctuation is True:
            for p in punctuation:
                text = text.replace(p, f' {p} ')
        if sanitize_white_space is True:
            text = text.replace('\n', ' ')
            text = text.replace('\t', ' ')
            text = text.replace('\r', ' ')
            text = text.replace('\f', ' ')
            text = text.replace('\v', ' ')
            to=""
            while to != text:
                to = text
                text = text.replace('  ', ' ')
        return text

    def get_random_sample(self, length, weighted=True):
        """ Return index idx and random sample of `length` chars from text[idx] the Text_Dataset.
        
        :param length: number of characters to return
        :param sanitize_white_space: If True, white space is replaced by a single space.
        :param separate_punctuation: If True, punctuation is separated from words.
        :param preserve_case: If True, the case of the text is preserved.
        :param weighted: If True, the probability of a text is weighted by its calculated 'probability_weight' attribute.
        :return: tuple (idx of text used for sampling, string of length `length` sampled from the Text_Dataset)
        """
        idx = self._get_random_text_index(weighted)
        text = self.text_list[idx]['text']
        if len(text) < length:
            sample = text
            while len(sample) < length:
                sample += ' '
        else:
            pos = random.randint(0, len(text) - length)
            sample = text[pos:pos+length]
        return (idx, sample)

    def _word_splitter(self, text, word_separator=' '):
        tokens=text.split(word_separator)
        return tokens

    def _every_ngram(self, text, n):
        ngrams = [text[i:i+j+1] for i in range(len(text)) for j in range(0, min(len(text)-i,m))]
        return ngrams

    def _weight_grams(self, ngrams, max_weight=1e12):
        eg_dict=Counter(ngrams)
        return sorted([(''.join(l), 1e12 if len(l)==1 else len(l)*eg_dict[l]) for l in eg_dict.keys()], key=lambda x: x[1], reverse=True) 

    def init_tokenizer(self, tokenizer='word', max_ngrams=5, word_separator=' ', max_tokens=5000):
        """ Initialize the tokenizer with the text_list.
        
        :param tokenizer: 'word', 'char', or 'ngram'
        :param max_ngrams: (ngram only) maximum n-gram length
        :param word_separator: (word, ngram) character used to separate words, None for languages with no word separator
        :param max_tokens: (ngram only) maximum number of tokens to use
        """
        if tokenizer == 'word':
            self.tokenizer_type = 'word'
            self.w2i = {}
            self.i2w = {}
            self.w2i['<unk>'] = 0
            self.i2w[0] = '<unk>'
            self.w2i['<pad>'] = 1
            self.i2w[1] = '<pad>'
            self.w2i['<eos>'] = 2
            self.i2w[2] = '<eos>'
            self.w2i['<sos>'] = 3
            self.i2w[3] = '<sos>'
            for text in self.text_list:
                tokens = self._word_splitter(text['text'], word_separator=word_separator)
                for token in tokens:
                    if token not in self.w2i:
                        self.w2i[token] = len(self.w2i)
                        self.i2w[len(self.w2i)-1] = token
            self.word_tokenizer_init=True
        elif tokenizer == 'char':
            self.tokenizer_type = 'char'
            self.i2c = {}
            self.c2i = {}
            self.c2i['␚'] = 0  # Unicode SUBSTITUTE for 'unknown'
            self.i2c[0] = '␚'
            self.c2i['␠'] = 1  # Unicode SPACE for 'pad'
            self.i2c[1] = '␠'
            self.c2i['␃'] = 2  # Unicode END OF TEXT for 'eos'
            self.i2c[2] = '␃'
            self.c2i['␂'] = 3  # Unicode START OF TEXT for 'sos'
            self.i2c[3] = '␂'
            for text in self.text_list:
                tokens = list(text['text'])
                unique_chars = set(tokens)
                for c in unique_chars:
                    if c not in self.c2i:
                        ind = len(self.i2c)
                        self.c2i[c] = ind
                        self.i2c[ind] = c
            self.char_tokenizer_init=True
        elif tokenizer == 'ngram':
            self.tokenizer_type = 'ngram'
            self.word_separator = word_separator
            self.max_ngrams = max_ngrams
            corpus=""
            for text in self.text_list:
                corpus+=text['text']
            if word_separator is not None:
                self.word_list=self.corpus.split(word_separator)
            else:
                self.word_list=None
            if self.word_list is None:
                ngrams=self._every_ngram(corpus,max_len=max_ngrams)
                self.ngrams_list = self.weight_grams(ngrams)
            else:
                ngrams=[]
                for word in self.word_list:
                    ngrams += self._every_ngrams(word, max_len=max_ngrams)
                self.ngrams_list = self.weight_ngrams(ngrams)
            if max_tokens is not None:
                if len(self.grams_list)>max_tokens:
                    self.grams_list=self.grams_list[:max_tokens]
            self.t2i={t[1][0]: t[0] for t in enumerate(self.grams_list)}
        else:
            self.log.error(f"Unknown tokenizer {tokenizer}")
            raise ValueError(f"Unknown tokenizer {tokenizer}")

    def tokenize(self, text):
        """ Tokenize a text.
        
        :param text: text to tokenize
        :return: list of tokens """
        tokens = []
        if self.tokenizer_type == 'word':
            if self.word_tokenizer_init is False:
                self.init_tokenizer(tokenizer)
            tokens = self._word_splitter(text)
        elif self.tokenizer_type == 'char':
            if self.char_tokenizer_init is False:
                self.init_tokenizer(tokenizer)
            tokens = list(text)
        elif self.tokenizer_type == 'ngram':
            if self.ngram_tokenizer_init is False:
                self.init_tokenizer(tokenizer)
            if self.word_separator is not None:
                wrd_list=text.split(self.word_separator)
                for word in wrd_list:
                    while len(word)>0:
                        mx = min(self.max_ngrams,len(word))            
                        for si in range(mx,0,-1):
                            tk=word[:si]
                            if tk in self.t2i:
                                ind = self.t2i[tk]
                                word = word[si:]
                                tokens.append(ind)
                                break
                        if len(word)>0:
                            if word[0] not in self.t2i:
                                tokens.append(-2)  # unknown encounter
                                word=word[1:]  # throw away one char
                    tokens.append(-1)  # separator
                if len(wrd_list)>1:
                    tokens=tokens[:-1]  # remove last seperator
            else:
                self.log.error("not implemented")
                print("Not yet implemented")
        else:
            self.log.error(f"Unknown tokenizer {self.tokenizer_type}")
            raise ValueError(f"Unknown tokenizer {self.tokenizer_type}")
        return tokens

    def encode(self, text):
        """ Encode a text.
        
        :param text: text to encode
        :return: list of encoded tokens """
        tokens = self.tokenize(text)
        if self.tokenizer_type == 'word':
            encoded = [self.w2i[token] if token in self.w2i else self.w2i['<unk>'] for token in tokens]
        elif self.tokenizer_type == 'char':
            encoded = [self.c2i[token] if token in self.c2i else self.c2i['␚'] for token in tokens]
        else:
            self.log.error(f"Unknown tokenizer {self.tokenizer_type}")
            raise ValueError(f"Unknown tokenizer {self.tokenizer_type}")
        return encoded

    def decode(self, encoded):
        """ Decode a list of encoded tokens.
        
        :param encoded: list of encoded tokens
        :return: text """
        if self.tokenizer_type == 'word':
            decoded = [self.i2w[token]+'' if token in self.i2w else '<unk>' for token in encoded]
            decoded_text = ' '.join(decoded)
        elif self.tokenizer_type == 'char':
            decoded = [self.i2c[token] if token in self.i2c else '␚' for token in encoded]
            decoded_text = ''.join(decoded)
        else:
            self.log.error(f"Unknown tokenizer {self.tokenizer_type}")
            raise ValueError(f"Unknown tokenizer {self.tokenizer_type}")
        return decoded_text

    def get_random_char_tokenized_sample_pair(self, length):
        """ Get a random tokenized sample of the dataset.
        
        :param length: length of the sample
        :return: tuple (X, y) encoded sample
        """
        _, sample = self.get_random_sample(length+1)
        e_sample = self.encode(sample)
        X = e_sample[:-1]
        y = e_sample[1:]
        return X, y

    def init_getitem(self, sample_type='chargen', sample_length=80, content_stepping=10):
        """ Initialize the __getitem__ and __len__ methods.

        This method needs to be called before using len() or index-access of the dataset.

        This method determines how the dataset is partitioned into records, and what kind
        of encoding is returned on index-access.

        .. code-block:: python

            from ml_indie_tools.Text_Dataset import TextDataset
            tl = [{'author':'nobody', 'title':'some title', 'language':'english', 'text':'some text'},
                  {'author':'nobody', 'title':'some title 2', 'language':'english', 'text':'some more text'}]
            td = Text_Dataset(tl)
            td.init_getitem(sample_type='chargen', sample_length=4, content_stepping=2)
            print(len(td))
            print(td[0])
            # Output: 12 and ('some', 'ome ')

        :param sample_type: 'chargen': generate a pair of text or length sample_length, shifted by one letter, or 'chargen_encoded', same but encoded. 'chargen_single_encoded' just returns a single encoded string X.
        :param sample_length: length of a sample
        :param content_stepping: number of characters to skip between each sample
        :return: on 'chargen[_encoded]': X, y [encoded] strings of sample_length, y shifted by one letter; for 'chargin_single_encoded' just encoded X.
        """

        self.getitem_sample_type = sample_type
        self.getitem_sample_length = sample_length
        self.getitem_content_stepping = content_stepping
        leng=0
        rec=0
        if sample_type=='chargen' or sample_type=='chargen_encoded' or sample_type=='chargen_single_encoded':
            for ind in range(0, len(self.text_list)):
                len_text = len(self.text_list[ind]['text'])
                rec_text = (len_text-content_stepping+1)//content_stepping + 1
                self.text_list[ind]['records']=rec_text
                leng += len_text
                rec += rec_text
            self.getitem_length = leng
            self.getitem_records = rec
            self.getitem_init = True
        else:
            self.getitem_length = 0
            self.getitem_records = 0
            print(f"init_getitem: unknown sample_type {sample_type}")

    def __len__(self):
        """ Get the length of the dataset.

        Note that this length depends on the initialization via :ref:`~Text_Dataset.Text_Dataset.init_getitem`.

        :return: length of the dataset (mode dependent) 
        """
        if self.getitem_init is False:
            print("init_getitem must be called before __len__")
            return None
        return self.getitem_records

    def _getitem_chargen(self, index):
        if index<0:
            if index<(-self.getitem_records):
                raise IndexError(f"index {index} out of range")
            else:
                index += self.getitem_records
        if index >= self.getitem_records:
            raise IndexError(f"index {index} out of range")
        cur_rec = 0
        for text in self.text_list:
            rec = text['records']
            if cur_rec+rec > index:
                rel_rec = index - cur_rec
                pos = rel_rec*self.getitem_content_stepping
                if self.getitem_sample_type=='chargen_encoded' or self.getitem_sample_type=='chargen':
                    sample = text['text'][pos:pos+self.getitem_sample_length+1]
                    while len(sample) < self.getitem_sample_length+1:
                        sample += ' '
                    X_text = sample[:-1]
                    y_text = sample[1:]
                    return X_text, y_text
                elif self.getitem_sample_type=='chargen_single_encoded':
                    sample = text['text'][pos:pos+self.getitem_sample_length]
                    while len(sample) < self.getitem_sample_length:
                        sample += ' '
                    X_text = sample
                    return X_text
                else:
                    print(f"_getitem_chargen: unknown sample_type {self.getitem_sample_type}")
                    return None
            cur_rec += rec
            
        print("Internal error in __getitem__")
        raise ValueError("Internal error in __getitem__")

    def __getitem__(self, index):
        """ Get a sample from the dataset.

        Format of the returned sample depends on :ref:`~Text_Dataset.Text_Dataset.init_getitem`.

        :param index: index of the sample
        :return:
        """
        if self.getitem_init is False:
            print("init_getitem must be called before __getitem__")
            raise ValueError("init_getitem must be called before __getitem__")
        if self.getitem_sample_type == 'chargen':
            return self._getitem_chargen(index)
        elif self.getitem_sample_type == 'chargen_encoded':
            X, y = self._getitem_chargen(index)
            X = self.encode(X)
            y = self.encode(y)
            return X, y
        elif self.getitem_sample_type == 'chargen_single_encoded':
            X = self._getitem_chargen(index)
            X = self.encode(X)
            return X
        else:
            self.log.error(f"Unknown getitem sample_type {self.getitem_sample_type}")
            raise ValueError(f"Unknown getitem sample_type {self.getitem_sample_type}")

    def _display_colored_html(self, textlist, dark_mode=False, display_ref_anchor=True, pre='', post=''):
        """ Internal function to display text and citation references in HTML. """
        bgcolorsWht = ['#d4e6e1', '#d8daef', '#ebdef0', '#eadbd8', '#e2d7d5', '#edebd0',
                    '#ecf3cf', '#d4efdf', '#d0ece7', '#d6eaf8', '#d4e6f1', '#d6dbdf',
                    '#f6ddcc', '#fae5d3', '#fdebd0', '#e5e8e8', '#eaeded', '#A9CCE3']
        bgcolorsDrk = ['#342621','#483a2f', '#3b4e20', '#2a3b48', '#324745', '#3d3b30',
                    '#3c235f', '#443f4f', '#403c37', '#463a28', '#443621', '#364b5f',
                    '#264d4c', '#2a3553', '#3d2b40', '#354838', '#3a3d4d', '#594C23']
        if dark_mode is False:
            bgcolors=bgcolorsWht
        else:
            bgcolors=bgcolorsDrk
        out = ''
        for txt, ind in textlist:
            txt = txt.replace('\n', '<br>')
            if ind == 0:
                out += txt
            else:
                if display_ref_anchor is True:
                    anchor="<sup>[" + str(ind) + "]</sup>"
                else:
                    anchor=""
                out += "<span style=\"background-color:"+bgcolors[ind % 16]+";\">" + \
                       txt + "</span>"+ anchor
        display(HTML(pre+out+post))

    def source_highlight(self, ref_txt, min_quote_size=10, dark_mode=False, display_ref_anchor=True):
        """ Analyse which parts of `ref_txt` are cited from the texts in the Text_Dataset.
        
        Note: this function requires a jupyter notebook in order to display HTML with markup.
        
        :param ref_txt: the reference text to be analysed for plagiarised parts
        :param min_quote_size: minimum size of a quote to be considered plagiarised
        :param dark_mode: if True, the background colors will be dark, otherwise white
        :param display_ref_anchor: if True, the reference text will be displayed with a reference anchor
        """
        ref_tx = ref_txt
        out = []
        qts = []
        txsrc = [("Sources: ", 0)]
        sc = False
        noquote = ''
        while len(ref_tx) > 0:  # search all library files for quote 'txt'
            mxQ = 0
            mxI = 0
            mxN = ''
            found = False
            for text in self.text_list:  # find longest quote in all texts
                p = min_quote_size
                if p <= len(ref_tx) and ref_tx[:p] in text['text']:
                    p = min_quote_size + 1
                    while p <= len(ref_tx) and ref_tx[:p] in text['text']:
                        p += 1
                    if p-1 > mxQ:
                        mxQ = p-1
                        mxI = text['index']
                        mxN = f"{text['author']}: {text['title']}"
                        found = True
            if found:  # save longest quote for colorizing
                if len(noquote) > 0:
                    out.append((noquote, 0))
                    noquote = ''
                out.append((ref_tx[:mxQ], mxI))
                ref_tx = ref_tx[mxQ:]
                if mxI not in qts:  # create a new reference, if first occurence
                    qts.append(mxI)
                    if sc:
                        txsrc.append((", ", 0))
                    sc = True
                    txsrc.append((mxN, mxI))
            else:
                noquote += ref_tx[0]
                ref_tx = ref_tx[1:]
        if len(noquote) > 0:
            out.append((noquote, 0))
            noquote = ''
        self._display_colored_html(out, dark_mode=dark_mode, display_ref_anchor=display_ref_anchor)
        if len(qts) > 0:  # print references, if there is at least one source
            self._display_colored_html(txsrc, dark_mode=dark_mode, display_ref_anchor=display_ref_anchor, pre="<small><p style=\"text-align:right;\">",
                                     post="</p></small>")
