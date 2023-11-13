import logging
import os
import uuid


class Folder_Dataset:
    """A class to access and search text documents from a folder."""

    def __init__(self):
        self.log = logging.getLogger("FolderTextLib")
        self.records = []
        self.index = 1

    def load_index(
        self,
        folder_path,
        parse_metadata=True,
        default_language=None,
        default_author=None,
        use_aliases=False,
    ):
        """This function loads the text files from the folder.

        Load index is additive, new texts are added to already imported texts.

        This loads the text files from a folder and creates some metadata from the filename.
        If `parse_metadata` is `True`, the filename format is expected to be: "Title - Author - Language.txt"

        :param folder_path: Path to a folder containing text files (.txt)
        :param parse_metadata: On True, filename must adhere to `title - author - language.txt` format.
        :param default_language: If language is not given via parse_metadata, use None or "English" etc.
        :param default_author: If author is not given via parse_metadata, use None or a specific author that applies to all texts.
        :param use_aliases: If True, documents are not referenced by filename (containing title and author),
        but by their numeric aliases, thus providing privacy.
        """
        folder_path = os.path.expanduser(folder_path)
        cur_index = 0
        if os.path.exists(folder_path) is False:
            raise FileNotFoundError("Folder not found at " + folder_path)
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # print(file)
                if file.endswith(".txt"):
                    components = file.split(" - ")
                    if len(components) == 3:
                        title = components[0]
                        author = components[1]
                        language = components[2].replace(".txt", "")
                    elif len(components) == 2:
                        title = components[0]
                        author = components[1].replace(".txt", "")
                        language = default_language
                    else:
                        author = default_author
                        language = default_language
                    filename = os.path.join(root, file)
                    # get a unique ID for the book using a crc from the filename
                    ebook_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, filename))
                    rec = {
                        "ebook_id": ebook_id,
                        "author": author,
                        "language": language,
                        "title": title,
                        "filename": filename,
                    }
                    if use_aliases is True:
                        rec["alias"] = f"FL{index}"

                    with open(filename, "r", encoding="utf-8") as f:
                        rec["text"] = f.read()
                    self.records += [rec]
                    index = index + 1
                    cur_index = cur_index + 1
        self.log.info(
            f"Loaded {cur_index} records from folder, grand total is now {len(self.records)} records."
        )
        return

    def get_records(self):
        """Return the imported text records"""
        return self.records

    def search(self, search_dict):
        """Search for book record with key specific key values
        For a list of valid keys, use `get_record_keys()`
        Standard keys are: `ebook_id`, `author`, `language`, `title`

        *Note:* :func:`~Folder_Dataset.Folder_Dataset.load_index` needs to be called once before this function can be used.

        Example: `search({"title": ["philosoph","phenomen","physic","hermeneu","logic"], "language":"english"})`
        Find all books whose titles contain at least one of the keywords, language english. Search keys can either be
        search for a single keyword (e.g. english), or an array of keywords.

        :returns: list of records"""
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
