import random
import logging
from collections import Counter

try:
    from IPython.core.display import display, HTML
except ImportError:
    pass


class Text_Dataset:
    """Initialize the Text_Dataset with a list of texts.

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

    :param text_list: list of text-records of the form: {'author': 'author', 'title': 'title', 'language': 'some-language',
    'text': 'the-long-text'}. Optinal parameters: 'weight': 1.0
    :param sanitize_white_space: If True, white space is replaced by a single space.
    :param separate_punctuation: If True, punctuation is separated from words.
    :param preserve_case: If True, the case of the text is preserved.
    """

    def __init__(
        self,
        text_list,
        sanitize_white_space=False,
        separate_punctuation=False,
        preserve_case=True,
    ):
        self.log = logging.getLogger("Datasets")
        self.text_list = []
        self.index = 1
        self.word_tokenizer_init = False
        self.char_tokenizer_init = False
        self.ngram_tokenizer_init = False
        self.getitem_init = False
        self.tokenizer_type = None
        self.special_words = ["<unk>", "<pad>", "<sos>", "<eos>", "<wsep>", "<subst>"]

        req_attrs = ["title", "author", "language", "text"]
        for ind in range(0, len(text_list)):
            valid = True
            miss = []
            for attr in req_attrs:
                if attr not in text_list[ind]:
                    valid = False
                    miss.append(attr)
            if valid is False:
                self.log.error(f"Missing attribute(s) {miss} in text[{ind}], skipping")
                continue
            text = text_list[ind]
            text["index"] = self.index
            text["text"] = self.filter_text(
                text["text"],
                sanitize_white_space=sanitize_white_space,
                separate_punctuation=separate_punctuation,
                preserve_case=preserve_case,
            )
            self.index += 1
            self.text_list.append(text)
        lt = len(self.text_list)
        if lt == 1:
            self.log.info(f"Loaded {lt} text")
        else:
            self.log.info(f"Loaded {lt} texts")
        self._calc_probability_weights()

    def _calc_probability_weights(self):
        prs = 0
        for text in self.text_list:
            if "weight" in text:
                w = text["weight"]
            else:
                w = 1.0
            pr = len(text["text"]) * w
            prs = prs + pr
            text["probability_weight"] = pr
        for text in self.text_list:
            text["probability_weight"] = text["probability_weight"] / prs
        self.tidx = []
        self.tcum = []
        tc = 0
        for idx in range(0, len(self.text_list)):
            self.tidx.append(idx)
            text = self.text_list[idx]
            self.tcum.append(text["probability_weight"] + tc)
            tc = self.tcum[-1]

    def _get_random_text_index(self, weighted=True):
        """Return a random text index from the Text_Dataset.

        :param weighted: If True, the probability of a text is weighted by its calculated 'probability_weight' attribute.
        :return: a random text index
        """
        if weighted is True:
            return random.choices(self.tidx, self.tcum)[0]
        else:
            return random.choice(self.tidx)

    def filter_text(
        self,
        text,
        sanitize_white_space=False,
        separate_punctuation=False,
        preserve_case=True,
    ):
        """Filter a text.

        :param text: text to filter
        :param sanitize_white_space: If True, white space is replaced by a single space.
        :param separate_punctuation: If True, punctuation is separated from words.
        :param preserve_case: If True, the case of the text is preserved.
        :return: filtered text
        """
        if preserve_case is False:
            text = text.lower()
        punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        if separate_punctuation is True:
            for p in punctuation:
                text = text.replace(p, f" {p} ")
        if sanitize_white_space is True:
            text = text.replace("\n", " ")
            text = text.replace("\t", " ")
            text = text.replace("\r", " ")
            text = text.replace("\f", " ")
            text = text.replace("\v", " ")
            to = ""
            while to != text:
                to = text
                text = text.replace("  ", " ")
        return text

    def _word_splitter(self, text, word_separator=" "):
        tokens = text.split(word_separator)
        return tokens

    def _every_ngram(self, text, max_len, add_special_words=True):
        """Return all ngrams of length 1..max_len from text.

        :param text: text to extract ngrams from
        :param max_len: maximum length of ngrams to extract
        :param add_special_words: If True, special words ('<unk>' etc.) are added to the ngrams.
        """
        ngrams = [
            text[i : i + j + 1]
            for i in range(len(text))
            for j in range(0, min(len(text) - i, max_len))
        ]
        if add_special_words is True:
            ngrams = self.special_words + ngrams
        return ngrams

    def _weight_ngrams(self, ngrams, max_weight=1e12):
        """Weight ngrams by their length and frequency. Ngrams of length==1 and special words ('<unk>' etc.) are
        weigted by max_weight, since we need the full character set and the special words to be
        included in the ngram collection.

        :param ngrams: ngrams to weight
        :param max_weight: maximum weight of ngrams, used to marked 'preferred' ngrams: ngrams of length==1 and special words ('<unk>' etc.)
        :return: ngrams with weights in reverse sorted order (highest weight first)
        """
        eg_dict = Counter(ngrams)
        return sorted(
            [
                (
                    "".join(lk),
                    max_weight
                    if len(lk) == 1 or lk in self.special_words
                    else len(lk) * eg_dict[lk],
                )
                for lk in eg_dict.keys()
            ],
            key=lambda x: x[1],
            reverse=True,
        )

    def init_tokenizer(
        self, tokenizer="word", max_ngrams=5, word_separator=None, max_tokens=5000
    ):
        """Initialize the tokenizer with the text_list.

        The character tokenizer simply splits any language text into glyphs.

        A word tokenizer works for languages that have the concept of word separators (as in English,
        ' ' space). Word tokenizers are English-centric algorithms, since many other languages either
        can compose words into larger words with no separators, words being more similar to English
        syllables in that case, or have no word separators at all (as in Chinese), or have syllable
        separators (as in Tibetan, which has also a second kind of separators for sentence-fragments). For multi-language applications, use ngram tokenizers or character
        tokenizers.

        The ngram tokenizer can either first split text into words using a word_speparator
        and then extracting all possible ngrams, or it can extract all possible ngrams directly
        without any word-splitting, which is the default behavior with word_separator=None since it
        works for all languages classes.

        For ngrams, the tokenizer can extract ngrams of length 1..max_ngrams, and selects the top most
        used ngrams with upper limit max_tokens. The optimum for max_ngrams is usually around 3-6, and
        max_tokens should be significantly higher than the number of unique glyphs in the text_list.
        Using word_separator=None is usually significantly better than using a word_separator for ngrams.

        :param tokenizer: 'word', 'char', or 'ngram'
        :param max_ngrams: (ngram only) maximum n-gram length
        :param word_separator: (word, ngram) character used to separate words, default None, which amounts to ' ' (space) for word and no word-splitting for ngram.
        :param max_tokens: (ngram only) maximum number of tokens to use
        """
        if tokenizer == "word":
            self.tokenizer_type = "word"
            self.w2i = {}
            self.i2w = {}
            for ind, tok in enumerate(self.special_words):
                self.w2i[tok] = ind
                self.i2w[ind] = tok
            for text in self.text_list:
                tokens = self._word_splitter(
                    text["text"], word_separator=word_separator
                )
                for token in tokens:
                    if token not in self.w2i:
                        self.w2i[token] = len(self.w2i)
                        self.i2w[len(self.w2i) - 1] = token
            self.word_tokenizer_init = True
        elif tokenizer == "char":
            self.tokenizer_type = "char"
            self.i2c = {}
            self.c2i = {}
            self.c2i["␚"] = 0  # Unicode SUBSTITUTE for 'unknown'
            self.i2c[0] = "␚"
            self.c2i["␠"] = 1  # Unicode SPACE for 'pad'
            self.i2c[1] = "␠"
            self.c2i["␃"] = 2  # Unicode END OF TEXT for 'eos'
            self.i2c[2] = "␃"
            self.c2i["␂"] = 3  # Unicode START OF TEXT for 'sos'
            self.i2c[3] = "␂"
            for text in self.text_list:
                tokens = list(text["text"])
                unique_chars = set(tokens)
                for c in unique_chars:
                    if c not in self.c2i:
                        ind = len(self.i2c)
                        self.c2i[c] = ind
                        self.i2c[ind] = c
            self.char_tokenizer_init = True
        elif tokenizer == "ngram":
            self.t2i = {}
            self.i2t = {}
            self.tokenizer_type = "ngram"
            self.word_separator = word_separator
            self.max_ngrams = max_ngrams
            self.log.info(
                f"Extracting ngrams of length 1..{max_ngrams} from text_list, selecting {max_tokens} most used ngrams."
            )
            corpus = ""
            for text in self.text_list:
                corpus += text[
                    "text"
                ]  # TODO: This generates ngrams across text-bordes, should be changed at some point.
            if word_separator is not None:
                self.word_list = corpus.split(word_separator)
            else:
                self.word_list = None
            if self.word_list is None:
                ngrams = self._every_ngram(corpus, max_len=max_ngrams)
                ngrams_list = self._weight_ngrams(ngrams)
            else:
                ngrams = []
                for word in self.word_list:
                    ngrams += self._every_ngram(word, max_len=max_ngrams)
                ngrams_list = self._weight_ngrams(ngrams)
            if max_tokens is not None:
                if len(ngrams_list) > max_tokens:
                    ngrams_list = ngrams_list[:max_tokens]
            self.t2i = {t[1][0]: t[0] for t in enumerate(ngrams_list)}
            del ngrams_list
            self.i2t = {t[1]: t[0] for t in self.t2i.items()}
            self.ngram_tokenizer_init = True
        else:
            self.log.error(f"Unknown tokenizer {tokenizer}")
            raise ValueError(f"Unknown tokenizer {tokenizer}")
            return
        self.log.info("Encoding text corpora as ngrams.")
        for text in self.text_list:
            self.log.info(f"Encoding text {text['title']}...")
            text["text_encoded"] = self.encode(text["text"])
        self.log.info("Encoding text corpora as ngrams done.")

    def tokenize(self, text):
        """Tokenize a text.

        :param text: text to tokenize
        :return: list of tokens"""
        tokens = []
        if self.tokenizer_type == "word":
            if self.word_tokenizer_init is False:
                self.init_tokenizer(tokenizer="word")
            tokens = self._word_splitter(text)
        elif self.tokenizer_type == "char":
            if self.char_tokenizer_init is False:
                self.init_tokenizer(tokenizer="char")
            tokens = list(text)
        elif self.tokenizer_type == "ngram":
            if self.ngram_tokenizer_init is False:
                self.init_tokenizer(tokenizer="ngram")
            if self.word_separator is not None:
                wrd_list = text.split(self.word_separator)
                for word in wrd_list:
                    while len(word) > 0:
                        is_special = False
                        for st in self.special_words:
                            if word.startswith(st):
                                tokens.append(self.t2i[st])
                                word = word[len(st) :]
                                is_special = True
                                break
                        if is_special is True:
                            continue
                        mx = min(self.max_ngrams, len(word))
                        for si in range(mx, 0, -1):
                            tk = word[:si]
                            if tk in self.t2i:
                                ind = self.t2i[tk]
                                word = word[si:]
                                tokens.append(ind)
                                break
                        if len(word) > 0:
                            if word[0] not in self.t2i:
                                tokens.append(self.t2i["<unk>"])  # unknown encounter
                                word = word[1:]  # throw away one char
                    tokens.append(self.t2i["<wsep>"])  # word separator
                if len(wrd_list) > 1:
                    tokens = tokens[:-1]  # remove last seperator
            else:
                while len(text) > 0:
                    is_special = False
                    for st in self.special_words:
                        if text.startswith(st):
                            tokens.append(self.t2i[st])
                            text = text[len(st) :]
                            is_special = True
                            break
                    if is_special is True:
                        continue
                    mx = min(self.max_ngrams, len(text))
                    for si in range(mx, 0, -1):
                        tk = text[:si]
                        if tk in self.t2i:
                            ind = self.t2i[tk]
                            text = text[si:]
                            tokens.append(ind)
                            break
                    if len(text) > 0:
                        if text[0] not in self.t2i:
                            tokens.append(self.t2i["<unk>"])
                            text = text[1:]
        else:
            self.log.error(f"Unknown tokenizer {self.tokenizer_type}")
            raise ValueError(f"Unknown tokenizer {self.tokenizer_type}")
        return tokens

    def encode(self, text):
        """Encode a text.

        :param text: text to encode
        :return: list of encoded tokens"""
        tokens = self.tokenize(text)
        if self.tokenizer_type == "word":
            encoded = [
                self.w2i[token] if token in self.w2i else self.w2i["<unk>"]
                for token in tokens
            ]
        elif self.tokenizer_type == "char":
            encoded = [
                self.c2i[token] if token in self.c2i else self.c2i["␚"]
                for token in tokens
            ]
        elif self.tokenizer_type == "ngram":
            encoded = tokens
        else:
            self.log.error(f"Unknown tokenizer {self.tokenizer_type}")
            raise ValueError(f"Unknown tokenizer {self.tokenizer_type}")
        return encoded

    def decode(self, encoded, mark_separator=False):
        """Decode a list of encoded tokens.

        :param encoded: list of encoded tokens
        :param mark_separator: (ngram only) if True, add a separator between tokens for debug
        :return: text"""
        if self.tokenizer_type == "word":
            decoded = [
                self.i2w[token] + "" if token in self.i2w else "<unk>"
                for token in encoded
            ]
            decoded_text = " ".join(decoded)
        elif self.tokenizer_type == "char":
            decoded = [
                self.i2c[token] if token in self.i2c else "␚" for token in encoded
            ]
            decoded_text = "".join(decoded)
        elif self.tokenizer_type == "ngram":
            dec = ""
            for ind in encoded:
                if ind == "<wsep>":  # Separator
                    dec += self.word_separator
                elif ind == "<unk>":  # Unknown token
                    dec += "<unk>"
                else:
                    dec += self.i2t[ind]
                    if mark_separator is True:
                        dec += "_"
            decoded_text = dec
        else:
            self.log.error(f"Unknown tokenizer {self.tokenizer_type}")
            raise ValueError(f"Unknown tokenizer {self.tokenizer_type}")
        return decoded_text

    def get_unique_token_count(self):
        """Get the number of unique tokens.

        :return: number of unique tokens"""
        if self.tokenizer_type == "word":
            return len(self.w2i)
        elif self.tokenizer_type == "char":
            return len(self.c2i)
        elif self.tokenizer_type == "ngram":
            return len(self.t2i)
        else:
            self.log.error(f"Unknown tokenizer {self.tokenizer_type}")
            raise ValueError(f"Unknown tokenizer {self.tokenizer_type}")

    def init_getitem(self, sample_type="text", sample_length=80, content_stepping=10):
        """Initialize the __getitem__ and __len__ methods.

        This method needs to be called before using len() or index-access of the dataset.

        This method determines how the dataset is partitioned into records, and what kind
        of encoding is returned on index-access.

        .. code-block:: python

            from ml_indie_tools.Text_Dataset import TextDataset
            tl = [{'author':'nobody', 'title':'some title', 'language':'english', 'text':'some text'},
                  {'author':'nobody', 'title':'some title 2', 'language':'english', 'text':'some more text'}]
            td = Text_Dataset(tl)
            td.init_getitem(sample_type='text', sample_length=4, content_stepping=2)
            print(len(td))
            print(td[0])
            # Output: 12 and ('some', 'ome ')

        The method of tokenization (char, word, ngram) for 'encoded' sample_type is determined by
        the tokenizer_type in call to init_tokenizer().

        :param sample_type: 'text' (text-string of length sample_length), 'encoded' (encoded sample_length tokens)
        :param sample_length: length of a sample (either character count (type text) or token count (type encoded))
        :param content_stepping: number of characters/tokens to skip between each sample
        """

        self.getitem_sample_type = sample_type
        self.getitem_sample_length = sample_length
        self.getitem_content_stepping = content_stepping
        len_tot = 0
        rec_tot = 0
        if sample_type not in ["text", "encoded"]:
            self.log.error(f"Unknown sample type {sample_type}")
            self.getitem_length = 0
            self.getitem_records = 0
            return None
        for ind in range(0, len(self.text_list)):
            if sample_type == "text":
                leni = len(self.text_list[ind]["text"])
            elif sample_type == "encoded":
                leni = len(self.text_list[ind]["text_encoded"])
            else:
                self.log.error(f"Unknown sample type {sample_type}")
                self.getitem_length = 0
                self.getitem_records = 0
                return None
            recs = (leni - content_stepping + 1) // content_stepping + 1
            self.text_list[ind]["records"] = recs
            len_tot += leni
            rec_tot += recs
        self.getitem_length = len_tot
        self.getitem_records = rec_tot
        self.getitem_init = True

    def __len__(self):
        """Get the length of the dataset.

        Note that this length depends on the initialization via :ref:`~Text_Dataset.Text_Dataset.init_getitem`.

        :return: length of the dataset (mode dependent)
        """
        if self.getitem_init is False:
            self.log.error("init_getitem must be called before __len__")
            return None
        return self.getitem_records

    def __getitem__(self, index):
        """Get a sample from the dataset.

        Format of the returned sample depends on :ref:`~Text_Dataset.Text_Dataset.init_getitem`.

        :param index: index of the sample
        :return:
        """
        if self.getitem_init is False:
            print("init_getitem must be called before __getitem__")
            raise ValueError("init_getitem must be called before __getitem__")
        if index < 0:
            if index < (-self.getitem_records):
                raise IndexError(f"index {index} out of range")
            else:
                index += self.getitem_records
        if index >= self.getitem_records:
            raise IndexError(f"index {index} out of range")
        cur_rec = 0
        for text in self.text_list:
            rec = text["records"]
            if cur_rec + rec > index:
                rel_rec = index - cur_rec
                pos = rel_rec * self.getitem_content_stepping
                if self.getitem_sample_type == "text":
                    sample = text["text"][pos : pos + self.getitem_sample_length]
                    while len(sample) < self.getitem_sample_length:
                        if self.tokenizer_type == "char":
                            sample += "␠"  # pad with ␠ character
                        elif self.tokenizer_type == "ngram":
                            sample += "<unk>"
                        elif self.tokenizer_type == "word":
                            sample += "<unk>"
                        else:
                            self.log.error(f"Unknown tokenizer {self.tokenizer_type}")
                            raise ValueError(f"Unknown tokenizer {self.tokenizer_type}")
                    return sample
                elif self.getitem_sample_type == "encoded":
                    sample = text["text_encoded"][
                        pos : pos + self.getitem_sample_length
                    ]
                    while len(sample) < self.getitem_sample_length:
                        if self.tokenizer_type == "char":
                            sample += [self.c2i["␠"]]  # pad with ␠ character
                        elif self.tokenizer_type == "ngram":
                            sample += [self.t2i["<unk>"]]
                        elif self.tokenizer_type == "word":
                            sample += [self.w2i["<unk>"]]
                        else:
                            self.log.error(f"Unknown tokenizer {self.tokenizer_type}")
                            raise ValueError(f"Unknown tokenizer {self.tokenizer_type}")
                    return sample
                else:
                    self.log.error(f"Unknown sample type {self.getitem_sample_type}")
                    return None
            cur_rec += rec
        print("Internal error in __getitem__")
        raise ValueError("Internal error in __getitem__")

    def get_random_item(self):
        """Get a random sample from the dataset.

        :return:
        """
        if self.getitem_init is False:
            print("init_getitem must be called before get_random_item")
            raise ValueError("init_getitem must be called before get_random_item")
        index = random.randint(0, self.getitem_records - 1)
        return self.__getitem__(index)

    def _display_colored_html(
        self, textlist, dark_mode=False, display_ref_anchor=True, pre="", post=""
    ):
        """Internal function to display text and citation references in HTML."""
        bgcolorsWht = [
            "#d4e6e1",
            "#d8daef",
            "#ebdef0",
            "#eadbd8",
            "#e2d7d5",
            "#edebd0",
            "#ecf3cf",
            "#d4efdf",
            "#d0ece7",
            "#d6eaf8",
            "#d4e6f1",
            "#d6dbdf",
            "#f6ddcc",
            "#fae5d3",
            "#fdebd0",
            "#e5e8e8",
            "#eaeded",
            "#A9CCE3",
        ]
        bgcolorsDrk = [
            "#342621",
            "#483a2f",
            "#3b4e20",
            "#2a3b48",
            "#324745",
            "#3d3b30",
            "#3c235f",
            "#443f4f",
            "#403c37",
            "#463a28",
            "#443621",
            "#364b5f",
            "#264d4c",
            "#2a3553",
            "#3d2b40",
            "#354838",
            "#3a3d4d",
            "#594C23",
        ]
        if dark_mode is False:
            bgcolors = bgcolorsWht
        else:
            bgcolors = bgcolorsDrk
        out = ""
        for txt, ind in textlist:
            txt = txt.replace("\n", "<br>")
            if ind == 0:
                out += txt
            else:
                if display_ref_anchor is True:
                    anchor = "<sup>[" + str(ind) + "]</sup>"
                else:
                    anchor = ""
                out += (
                    '<span style="background-color:'
                    + bgcolors[ind % 16]
                    + ';">'
                    + txt
                    + "</span>"
                    + anchor
                )
        display(HTML(pre + out + post))

    def source_highlight(
        self, ref_txt, min_quote_size=10, dark_mode=False, display_ref_anchor=True
    ):
        """Analyse which parts of `ref_txt` are cited from the texts in the Text_Dataset.

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
        noquote = ""
        while len(ref_tx) > 0:  # search all library files for quote 'txt'
            mxQ = 0
            mxI = 0
            mxN = ""
            found = False
            for text in self.text_list:  # find longest quote in all texts
                p = min_quote_size
                if p <= len(ref_tx) and ref_tx[:p] in text["text"]:
                    p = min_quote_size + 1
                    while p <= len(ref_tx) and ref_tx[:p] in text["text"]:
                        p += 1
                    if p - 1 > mxQ:
                        mxQ = p - 1
                        mxI = text["index"]
                        mxN = f"{text['author']}: {text['title']}"
                        found = True
            if found:  # save longest quote for colorizing
                if len(noquote) > 0:
                    out.append((noquote, 0))
                    noquote = ""
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
            noquote = ""
        self._display_colored_html(
            out, dark_mode=dark_mode, display_ref_anchor=display_ref_anchor
        )
        if len(qts) > 0:  # print references, if there is at least one source
            self._display_colored_html(
                txsrc,
                dark_mode=dark_mode,
                display_ref_anchor=display_ref_anchor,
                pre='<small><p style="text-align:right;">',
                post="</p></small>",
            )
