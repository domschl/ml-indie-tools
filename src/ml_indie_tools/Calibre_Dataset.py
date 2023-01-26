import logging
import os
import time
from xml.etree import ElementTree as ET


class Calibre_Dataset:
    """A class to access and search text documents from a Calibre library.

    :param library_path: Path to the Calibre library
    """

    def __init__(self, library_path, verbose=True):

        # old root, vanished: http://www.mirrorservice.org/sites/ftp.ibiblio.org/pub/docs/books/gutenberg
        self.log = logging.getLogger("CalibreLib")
        if os.path.exists(os.path.join(library_path, "metadata.db")) is False:
            raise FileNotFoundError("Calibre library not found at " + library_path)

        self.library_path = library_path
        self.verbose = verbose

    def load_index(self):
        """This function loads the Calibre library records that contain text-format books."""
        # Enumerate all txt-format books
        self.records = []
        for root, dirs, files in os.walk(self.library_path):
            for file in files:
                if file.endswith(".txt"):
                    # self.records += [os.path.join(root, file)]
                    # remove txt extension and add opf extension
                    opf_file = os.path.splitext(os.path.join(root, file))[0] + ".opf"
                    if os.path.exists(opf_file):
                        tree = ET.parse(opf_file)
                        title = tree.find(
                            ".//{http://purl.org/dc/elements/1.1/}title"
                        ).text
                        author = tree.find(
                            ".//{http://purl.org/dc/elements/1.1/}creator"
                        ).text
                        language = tree.find(
                            ".//{http://purl.org/dc/elements/1.1/}language"
                        ).text
        # xxx

    def search(self, search_dict):
        """Search for book record with key specific key values
        For a list of valid keys, use `get_record_keys()`
        Standard keys are: `ebook_id`, `author`, `language`, `title`

        *Note:* :func:`~Calibre_Dataset.Calibre_Dataset.load_index` needs to be called once before this function can be used.

        Example: `search({"title": ["philosoph","phenomen","physic","hermeneu","logic"], "language":"english"})`
        Find all books whose titles contain at least one of the keywords, language english. Search keys can either be
        search for a single keyword (e.g. english), or an array of keywords.

        :returns: list of records"""
        if not hasattr(self, "records") or self.records is None:
            self.log.debug("Index not loaded, trying to load...")
            self.load_index()
        frecs = []
        for rec in self.records:
            found = True
            for sk in search_dict:
                if sk not in rec:
                    found = False
                    break
                else:
                    skl = search_dict[sk]
                    if not isinstance(skl, list):
                        skl = [skl]
                    nf = 0
                    for skli in skl:
                        if skli.lower() in rec[sk].lower():
                            nf = nf + 1
                    if nf == 0:
                        found = False
                        break
            if found is True:
                frecs += [rec]
        return frecs

    def get_book(self, ebook_id: str):
        """Get a book record metadata and text by its ebook_id

        *Note:* :func:`~Calibre_Dataset.Calibre_Dataset.load_index` needs to be called once before this function can be used.

        :param ebook_id: ebook_id (String, since some IDs contain letters) of the book to be retrieved
        :returns: book record (dictionary with metadata and filtered text)
        """
        for rec in self.records:
            if rec["ebook_id"] == ebook_id:
                text, _, valid = self._load_book_ex(ebook_id)
                if text is None or valid is False:
                    self.log.Error(f"Download of book {ebook_id} failed!")
                    return None
                rec["text"] = self.filter_text(text)
                return rec
        return None
