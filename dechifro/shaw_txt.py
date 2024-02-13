#!/bin/python3
"""
Runs standard input through a part-of-speech tagger, then
translates to Shavian. This resolves most heteronyms, but
do still check the output for @ signs and fix them by hand.

Each line of a dictionary consists of an English word, a space,
a Shavian translation, and no comments. Special notations are:

^word 𐑢𐑻𐑛	word is a prefix
$word 𐑢𐑻𐑛	word is a suffix
Word 𐑢𐑻𐑛	always use a namer dot
word_ 𐑢𐑻𐑛	never use a namer dot
word_VB 𐑢𐑻𐑛	shave this way when tagged as a verb
word. 𐑢𐑻𐑛	shave this way when no suffix is present
word .𐑢𐑻𐑛	word takes no prefixes
word 𐑢𐑻𐑛.	word takes no suffixes
word 𐑢𐑻𐑛:	suffixes do not alter the root,
	      	e.g. "𐑑𐑾" or "𐑕𐑾" palatizing to "𐑖𐑩" or "𐑠𐑩".
word .		delete this word from the dictionary

Words are matched case-sensitive when possible, e.g. US/us,
WHO/who, Job/job, Nice/nice, Polish/polish.

shaw.py does not care about the order of dictionary entries.
shaw.c requries a highly specific order not described here.
"""
import importlib.resources
import re
import os
import sys
import html
from html.parser import HTMLParser

import pandas as pd

apostrophe = "'"  # whatever you want for apostrophe, e.g. "’" or ""
dot_entire_name = True
if os.path.exists("config.py"):
    from config import *

script = flair = spaCy = notags = False
dict = htags = {}
tokens = ["."]
units = {"ms": "𐑥𐑕", "bc": "𐑚𐑰𐑕𐑰", "psi": "𐑐𐑕𐑦", "pc": "𐑐𐑕", "mi": "𐑥𐑲"}
contr = ["'d", "'ll", "'m", "n't", "'re", "'s", "'ve"]
abbrev = [
    "abbr",
    "acad",
    "al",
    "alt",
    "apr",
    "assn",
    "at",
    "aug",
    "ave",
    "b",
    "c",
    "cf",
    "capt",
    "cent",
    "chm",
    "chmn",
    "co",
    "col",
    "comdr",
    "corp",
    "cpl",
    "d",
    "dec",
    "dept",
    "dist",
    "div",
    "dr",
    "ed",
    "esq",
    "est",
    "etc",
    "feb",
    "fl",
    "gen",
    "gov",
    "hon",
    "inc",
    "inst",
    "jan",
    "jr",
    "lat",
    "lib",
    "lt",
    "ltd",
    "mar",
    "mr",
    "mrs",
    "ms",
    "msgr",
    "mt",
    "mts",
    "mus",
    "nov",
    "oct",
    "pg",
    "phd",
    "pl",
    "pop",
    "pp",
    "prof",
    "pseud",
    "pt",
    "rev",
    "sept",
    "ser",
    "sgt",
    "sr",
    "st",
    "uninc",
    "univ",
    "vol",
    "vs",
    "wt",
]


# Remove diacritics from Latin letters, break up ligatures, and do nothing else.
def unaccent(str):
    map = "AAAAAA CEEEEIIIIDNOOOOO OUUUUY  aaaaaa ceeeeiiiidnooooo ouuuuy yAaAaAaCcCcCcCcDdDdEeEeEeEeEeGgGgGgGgHhHhIiIiIiIiIi  JjKkkLlLlLlLlLlNnNnNn   OoOoOo  RrRrRrSsSsSsSsTtTtTtUuUuUuUuUuUuWwYyYZzZzZz bBBb   CcDDDdd  EFfG  IIKkl  NnOOo  Pp     tTtTUuYVYyZz    255      Ǳǲǳ      AaIiOoUuUuUuUuUu AaAaÆæGgGgKkOoOo  j   Gg  NnAaÆæOoAaAaEeEeIiIiOoOoRrRrUuUuSsTt  HhNd  ZzAaEeOoOoOoOoYylntj  ACcLTsz  BU EeJjqqRrYy"
    lig = {
        "Æ": "AE",
        "æ": "ae",
        "Ǳ": "DZ",
        "ǲ": "Dz",
        "ǳ": "dz",
        "Ĳ": "Ij",
        "ĳ": "ij",
        "Ǉ": "LJ",
        "ǈ": "Lj",
        "ǉ": "lj",
        "Ǌ": "NJ",
        "ǋ": "Nj",
        "ǌ": "nj",
        "Œ": "OE",
        "œ": "oe",
        "Ƣ": "OI",
        "ƣ": "oi",
        "ß": "ss",
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "st",
        "ﬆ": "st",
    }
    ret = ""
    for char in str:
        n = ord(char)
        if n >= 0xC0 and n < 0x250 and map[n - 0xC0] != " ":
            char = map[n - 0xC0]
        if n >= 0x300 and n < 0x370:
            char = ""
        if char in lig:
            char = lig[char]
        ret += char
    return ret


def notrans(str):
    global htags
    if toki in htags:
        htags[toki] += str
    else:
        htags[toki] = str


def tokenize(str):
    global tokens, toki
    str = " " + unaccent(html.unescape(str)) + " "
    old = 0
    for i in range(1, len(str) - 1):
        new = 0
        if str[i].isalpha():
            new = 1
        if str[i].isdigit():
            new = 2
        if str[i] == " " and tokens[-1][0].isalpha() and str[i + 1].isalpha():
            new = 0
        if str[i] in "'." and str[i - 1].isalpha() and str[i + 1].isalpha():
            new = 1
        if str[i] in ",." and str[i - 1].isdigit() and str[i + 1].isdigit():
            new = 2
        if str[i] == "." and old == 1 and "." in tokens[-1]:
            new = 1  # U.S.A. keeps the dot
        if str[i] == "." and new == 0 and tokens[-1].lower() in abbrev:
            continue  # Dr. Mr. etc. lose it
        if old and old == new:
            tokens[-1] += str[i]
        else:
            for c in contr:  # break up contractions so PoS tagging works
                s = len(tokens[-1]) - len(c)
                if s < 1:
                    continue
                low = tokens[-1][s:].lower()
                if c == low:
                    tokens[-1] = tokens[-1][:s]
                    tokens.append(low)
            tokens.append(str[i])
            toki += 1
        old = new
        if (
                tokens[-1].isspace()
                or not tokens[-1].isprintable()
                or ord(tokens[-1][0]) | 15 == 0xFE0F
        ):
            toki -= 1  # Whitespace tokens break NLTK and variation
            notrans(tokens.pop())  # selectors break Flair. Move these to htags.


class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        global script
        out = "<" + tag
        for at in attrs:
            if at[0] == "charset":
                at = ("charset", "UTF-8")
            if at[0] == "content":
                at = ("content", "text/html; charset=UTF-8")
            out += " " + at[0]
            if type(at[1]) == str:
                out += '="' + at[1] + '"'
        out += ">"
        if tag == "noscript" or tag == "script" or tag == "style":
            script = True
        notrans(out)

    def handle_endtag(self, tag):
        global script
        notrans("</" + tag + ">")
        if tag == "noscript" or tag == "script" or tag == "style":
            script = False

    def handle_data(self, data):
        if script:
            notrans(data)
        else:
            tokenize(data)


# Search all the ways a word might appear in the dictionary
def lookup(word, pos):
    if not aflag and re.fullmatch("[WMDC]*[CLX]*[XVI]*", word) and word != "I":
        return word
    ret = ""
    low = word.lower()
    upp = word[0].upper() + low[1:]
    pos = "_" + pos
    if aflag & 2:
        list = [
            low + pos,
            low + pos[:3],
            word,
            word + "_",
            low,
            low + "_",
            low + "_NN",
            low + "_NNS",
            upp,
        ]
    else:
        list = [
            low + pos,
            low + pos[:3],
            word + ".",
            upp + ".",
            word,
            word + "_",
            low + ".",
            low,
            low + "_",
            low + "_NN",
            low + "_NNS",
            upp,
        ]
    for look in list:
        if look in dict:
            ret = dict[look]
            if aflag & 1 and ret[0] == "." or aflag & 2 and ret[-1] == ".":
                ret = ""
            ret = ret.replace(".", "")
            if not ret:
                continue
            if (
                    (word[0].isupper() or look[0].isupper())
                    and (look[-1] != "_" or aflag)
                    and not re.search("[A-Z]", ret)
            ):
                ret = "·" + ret
            break
    return ret


def suffix_split(inp, pos, adj):
    global aflag
    long = len(inp)
    root = lookup(inp, pos)
    if root:
        return ((long + adj) ** 2, root)
    low = inp.lower()
    best = (0, "")
    for split in range(2, long):
        suff = "$" + low[split:]
        if not suff in dict:
            continue
        if low[split] == low[split - 1]:
            if long - split == 1 or low[split] in "eos":
                continue
        if low[split:] == "es" and low[split - 1] not in "hiosuxz":
            continue
        if low[split:] == "ry" and low[split - 1] in "aeiouf":
            continue
        if low[split:] == "ha" and low[split - 1] in "cst":
            continue
        if low[split:] == "th" and low[split - 1] in "e":
            continue
        if low[split:] == "t" and low[split - 1] in "aeioust'":
            continue
        if low[split:] == "k" and low[split - 1] in "aceino":
            continue
        if low[split:] == "r" and low[split - 1] in "aeiou":
            continue
        if low[split:] == "m" and low[split - 1] in "eis":
            continue
        if low[split:] == "z" and low[split - 1] in "i":
            continue
        if (
                low[split:] == "n"
                and low[split - 1] in "eio"
                and low[split - 2] != low[split - 1]
        ):
            continue
        if (
                low[split:] == "ess"
                and low[split - 1] in "ln"
                and low[split - 2] == low[split - 1]
        ):
            continue
        suff = dict[suff]
        for pess in range(2):
            if (pess) or inp[split] == "'":
                word = inp[:split]
            elif low[split - 1] == "i" and low[split] not in "cfikmpsv":
                word = inp[: split - 1] + "y"
            elif (
                    low[split] in "aeiouy"
                    and low[split - 1] not in "aeio"
                    and low[split: split + 2] not in ["ub", "up"]
            ):
                if low[split - 1] == low[split - 2] and low[split - 1] not in "hsw":
                    word = inp[: split - 1]
                elif (
                        low[split - 1] in "cghlsuvz"
                        or low[split] == "e"
                        or low[split - 2] in "aeiousy"
                ) and (low[split - 1] not in "cg" or low[split] not in "aou"):
                    word = inp[:split] + "e"
                else:
                    continue
            elif low[split - 2: split] == "dg":
                word = inp[:split] + "e"
            else:
                continue
            aflag &= ~2
            if inp[split] != "'":
                aflag |= 2
            root = suffix_split(word, "UNK", split - len(word))
            score = (long - split + adj) ** 2 + root[0] if root[0] else 0
            if score:
                if low[split:] in ["call"]:
                    score = 1
                if low[split:] in [
                    "bed",
                    "can",
                    "cent",
                    "dance",
                    "den",
                    "ine",
                    "kin",
                    "one",
                    "pal",
                    "path",
                    "ster",
                    "wing",
                    "x",
                ]:
                    score = max(1, score - 9)
            if score <= best[0]:
                continue
            root = root[1]
            if (
                    low[split - 1] == "e"
                    and low[split - 2] not in "aegiou"
                    and low[split] in "aou"
                    and (split + 1 == long or low[split + 1] in "dlmnprstu")
                    and low[split:] not in ["arm", "out", "und", "up"]
            ):
                if root[-1] in "𐑦𐑰":
                    root = root[:-1]
                root += "𐑦"
            if (
                    root[-1] == "𐑓"
                    and suff[0] > "𐑗"
                    and word[-1] in "vw"
                    and low[split:] != "s"
            ):
                root = root[:-1] + "𐑝"
            if root[-2:] == "𐑩𐑤" and suff == "𐑦" and word[-2:] == "le":
                root = root[:-2] + "𐑤"
            if root[-3:] == "𐑟𐑩𐑥" and suff not in ["'", "𐑛:", "𐑟:", "𐑦𐑙"]:
                root = root[:-3] + "𐑟𐑥"
            if root[-2:] in ["𐑩𐑤", "𐑭𐑤", "𐑾𐑤"] and suff == "𐑦𐑑𐑦":
                mid = "𐑦" if root[-3] in "𐑖𐑗𐑠𐑡" or root[-2] == "𐑾" else ""
                root = root[:-2] + mid + "𐑨𐑤"
            mid = root[-1] + suff[0]
            if mid == "𐑦𐑩":
                mid = "𐑾"
            if mid == "𐑦𐑼":
                mid = "𐑽"
            if mid in ["𐑤𐑤", "𐑯𐑯"] and len(suff) < 3:
                mid = mid[0]
            best = (score, root[:-1] + mid + suff[1:])
    if long > 1 and low[-1] == low[-2] and low[-2] not in "aeiosu":
        aflag |= 2
        root = suffix_split(inp[:-1], "UNK", 0)
        if best[0] < root[0]:
            best = root
    if len(best[1]) > 1:
        word = best[1][:-2]
        end = best[1][-2:]
        if end in ["𐑛:", "𐑟:"]:
            tail = -1
            while word[tail] in ["'", ":"]:
                tail -= 1
            if word[tail] in {"𐑛:": "𐑑𐑛", "𐑟:": "𐑕𐑖𐑗𐑟𐑠𐑡"}[end]:
                word += "𐑩"
            elif word[tail] >= "𐑐" and word[tail] < "𐑘":
                end = chr(ord(end[0]) - 10) + ":"
        word += end
        if word[-4:] == "𐑒𐑩𐑤𐑦" and word[-5] in "𐑦𐑩":
            word = word[:-4] + "𐑒𐑤𐑦"
        if word[-1] == "𐑾":
            word += "0"
        if (
                word[-2] == "𐑾"
                and len(word) > 4
                and word[-3] in "𐑑𐑕𐑟"
                and word[-1] in "𐑕𐑤𐑯0"
        ):
            mid = "𐑖"
            if word[-3] == "𐑑":
                if word[-4] == "𐑕":
                    mid = "𐑗"
            elif word[-1] in "𐑯0" and word[-4] in "𐑰𐑱𐑴𐑵𐑷𐑻𐑿":
                mid = "𐑠"
            word = word[:-3] + mid + "𐑩" + word[-1]
        if word[-1] == "0":
            word = word[:-1]
        best = (best[0], word)
    return best


# aflag:
# "apostraphe_flag" : indicates apostraphe boundary.
# 1 if in a part followed by apostraphe
# 2 if in a part preceded by apostraphe


def prefix_split(word, pos, ms):
    global aflag
    best = suffix_split(word, pos, 0)
    if best[0] == len(word) ** 2:
        return best
    for split in range(len(word) - 2, ms, -1):
        pref = "^" + word[:split].lower()
        if not pref in dict:
            continue
        if word[: split + 1].lower() == "un":
            continue
        aflag = word[split - 1] != "'"
        root = prefix_split(word[split:], pos, 1)
        score = split ** 2 + root[0] if root[0] else 0
        if pref == "^la":
            score -= 4
        if score > best[0]:
            dot = "·" if word[0].isupper() else ""
            pref = dict[pref]
            if (
                    pref[-1] == root[1][0]
                    and pref[-1] in "𐑤𐑥𐑮𐑯"
                    and pref[-2] == "𐑦"
                    or pref == "𐑥𐑩𐑒"
                    and (root[1][0] == "𐑒" or root[1][:2] == "·𐑒")
            ):
                pref = pref[:-1]
            best = (score, pref + dot + root[1])
    return best


first = True
notags = True

with importlib.resources.files('dechifro').joinpath('amer.dict.xz').open('rb') as xz:
    df = pd.read_csv(xz, header=None, names=['key', 'val'], dtype=str, na_filter=False, compression='xz', sep=' ')
    for i, (key, val) in df.iterrows():
        if notags:
            key = re.sub("_[A-Z]+", "", key)
            if key + "_" in dict:
                key += "_"
        if first and key in dict:
            if val not in dict[key].split("@"):
                dict[key] += "@" + val
        else:
            dict[key] = val
        if not first:  # Allow extra dicts to force dotting
            low = key.lower()
            if low != key and low in dict:
                del dict[low]
        if val == ".":
            del dict[key]


def isolated_lookup(word, pos='UNK'):
    global aflag
    try:
        aflag = 0
        dot = word[0].isupper()
        root = prefix_split(word, pos, 0)
        if root[1]:
            tran = root[1].replace("'", apostrophe).replace(":", "")
            dot = dot or "·" in tran
            tran = tran.replace("·", "")
            if dot:
                tran = "·" + tran
            return tran
        else:
            return None
    except IndexError:
        print(f'Failed to convert {word!r}', file=sys.stderr)
        return None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:", sys.argv[0], "file1.dict file2.dict ...")
        exit()

    for fname in sys.argv[1:]:
        if fname == "-f":  # Use Flair for POS tagging
            save = sys.stdout
            sys.stdout = sys.stderr
            from flair.data import Sentence
            from flair.models import SequenceTagger

            flair = SequenceTagger.load("flair/pos-english-fast")
            sys.stdout = save
            continue
        if fname == "-s":  # Use spaCy for POS tagging
            import spacy

            spaCy = spacy.load("en_core_web_sm")
            continue
        if fname == "-n":
            notags = True
            continue
        first = False

    text = sys.stdin.read()
    text = re.sub("([ʻˈ‘’´`]|&(#8217|rsquo);)", "'", text)
    text = re.sub(r"\[([a-zA-Z])\]", r"\1", text)
    text = re.sub(r"\b[Dd][Ee]-", r"dee-", text)
    text = re.sub(r"\bw/o\b", r"without", text)
    text = re.sub(r"\bw/", r"with ", text)
    text = re.sub("[­​]", "", text)  # soft hyphen, zero-width space

    toki = 1
    parser = MyHTMLParser()
    parser.feed(text)

    tags = []
    if flair:  # Do part-of-speech tagging
        sen = []
        for tok in tokens + ["."]:
            sen.append(tok)
            if tok in [".", "?", "!"]:
                sen = Sentence(sen)
                flair.predict(sen)
                for tok in sen:
                    tags.append((tok.text, tok.get_label("pos").value))
                sen = []
        del tags[-1]
    elif spaCy:
        tags = [(w.text, w.tag_) for w in spaCy(spacy.tokens.Doc(spaCy.vocab, tokens))]
    elif notags:
        tags = [(tok, "UNK") for tok in tokens]
    else:
        import nltk

        tags = nltk.pos_tag(tokens)

    jtags = []  # Re-join broken contractions
    for token in tags:
        if token[0].lower() in contr:
            jtags[-1] = (jtags[-1][0] + token[0], jtags[-1][1] + "+" + token[1])
        else:
            jtags.append(token)

    out = []  # Translate to Shavian
    prev = (".", ".")
    toki = 1
    initial = maydot = True
    map = {"𐑑": "𐑑𐑵", "𐑓": "𐑓𐑹", "𐑝": "𐑩𐑝", "𐑯": "𐑨𐑯𐑛"}
    if False == 1:
        map.update({"𐑩": "𐑱", "𐑩𐑯": "𐑨𐑯", "𐑝": "𐑪𐑝"})
    else:
        map["𐑞"] = "𐑞𐑩"
    for token in jtags[1:]:
        if toki in htags:
            out.append(htags[toki])
            if htags[toki].lower().find("<div") + 1:
                initial = True
        toki += 1
        #  print (token, file=sys.stderr)
        if token[0] in [".", "?", "!", ":", '"', "“", "”"]:
            initial = True
        word = token[0]
        low = word.lower()
        befto = {"have": "𐑨𐑓", "has": "𐑨𐑕", "used": "𐑕𐑑", "unused": "𐑕𐑑", "supposed": "𐑕𐑑"}
        if prev[0] in befto and low == "to":
            if notags:
                out.extend(["@", befto[prev[0]], " "])
            elif prev != ("used", "VBN"):  # If "to" changes the meaning of the preceding
                for i in reversed(
                        range(len(out))
                ):  # word, it also changes the pronunciation.
                    if out[i] == tran:
                        break
                out[i] = tran[:-2] + befto[prev[0]]
        tran = word
        if word[0].isalpha():
            dot = word[0].isupper()
            if initial:
                initial = False
                if len(word) == 1 or word[1].islower():
                    word = word[0].lower() + word[1:]
            aflag = 0
            root = prefix_split(word, token[1], 0)
            if root[1]:
                tran = root[1].replace("'", apostrophe).replace(":", "")
            if prev[0][0].isdigit() and low in units:
                tran = units[low]
            if not False:
                dot = maydot and "·" in tran
            tran = tran.replace("·", "")
            if False and tran in map:
                tran = map[tran]
            if False == 1:
                if tran[-2:] in ["𐑩𐑤", "𐑩𐑥", "𐑩𐑯"]:
                    tran = tran[:-2] + tran[-1]
            if False == 2:
                tran = tran.replace("𐑣𐑒", "x").replace("𐑣𐑜", "ɣ").replace("𐑣𐑤", "ɬ")
            if False == 3:
                for tup in [
                    ("𐑦𐑹", "𐑦𐑷𐑮"),
                    ("𐑣𐑿", "𐑣𐑘𐑵"),
                    ("𐑒𐑢", "ᛢ"),
                    ("𐑕𐑗", "ᛥᛄ"),
                    ("𐑕𐑑", "ᛥ"),
                    ("𐑣𐑤", "ᚻ\u200dᛚ"),
                ]:
                    tran = tran.replace(tup[0], tup[1])
                tran = re.sub("([𐑦𐑰][𐑪𐑴𐑷]|𐑣[𐑒𐑘𐑜])", "ᛇ", tran)
            if False and low[-2:] not in ["id", "iz", "zz"]:
                for i in range(3, len(tran)):
                    if tran[i] == "𐑦" and (
                            tran[i + 1:] in ["", "𐑛", "𐑟"] or tran[i + 1] == "𐑦"
                    ):
                        tran = tran[:i] + "𐒀" + tran[i + 1:]
            if dot:
                tran = "·" + tran
                maydot = dot_entire_name
            else:
                maydot = True
        elif word != "-":
            maydot = True  # Names may contain hyphens
        out.append(tran)
        if low != " ":
            prev = (low, token[1])
    if toki in htags:
        out.append(htags[toki])
    out = "".join(out).replace("𐑲𐑟𐑱𐑖𐑩𐑯", dict["ization"][1:])
    if False:
        map = [
            [
                "𐐹",
                "𐐻",
                "𐐿",
                "𐑁",
                "𐑃",
                "𐑅",
                "𐑇",
                "𐐽",
                "𐐷",
                "𐑍",
                "𐐺",
                "𐐼",
                "𐑀",
                "𐑂",
                "𐑄",
                "𐑆",
                "𐑈",
                "𐐾",
                "𐐶",
                "𐐸",
                "𐑊",
                "𐑋",
                "𐐮",
                "𐐯",
                "𐐰",
                "𐐲",
                "𐐱",
                "𐐳",
                "𐐵",
                "𐐪",
                "𐑉",
                "𐑌",
                "𐐨",
                "𐐩",
                "𐐴",
                "𐐲",
                "𐐬",
                "𐐭",
                "𐑎",
                "𐐫",
                "𐐪𐑉",
                "𐐫𐑉",
                "𐐩𐑉",
                "𐐲𐑉",
                "𐐲𐑉",
                "𐐨𐑉",
                "𐐨𐐲",
                "𐑏",
                "𐐨",
            ],
            [
                "p",
                "t",
                "k",
                "f",
                "θ",
                "s",
                "ʃ",
                "tʃ",
                "j",
                "ŋ",
                "b",
                "d",
                "g",
                "v",
                "ð",
                "z",
                "ʒ",
                "dʒ",
                "w",
                "h",
                "l",
                "m",
                "ɪ",
                "ɛ",
                "æ",
                "ə",
                "ɒ",
                "ʊ",
                "aʊ",
                "ɑː",
                "r",
                "n",
                "iː",
                "eɪ",
                "aɪ",
                "ʌ",
                "əʊ",
                "u",
                "ɔɪ",
                "ɔː",
                "ɑːr",
                "ɔːr",
                "ɛər",
                "ɜːr",
                "ər",
                "ɪər",
                "ɪə",
                "ju",
                "i",
            ],
            [
                "ᛈ",
                "ᛏ",
                "ᛣ",
                "ᚠ",
                "ᚦ",
                "ᛋ",
                "ᛋᚳ",
                "ᚳ",
                "ᛄ",
                "ᛝ",
                "ᛒ",
                "ᛞ",
                "ᚸ",
                "ᚠ\u200dᚠ",
                "ᚦ",
                "ᛉ",
                "ᛉᚳ",
                "ᚷ",
                "ᚹ",
                "ᚻ",
                "ᛚ",
                "ᛗ",
                "ᛁ",
                "ᛖ",
                "ᚫ",
                "ᛟ",
                "ᚩ",
                "ᚢ",
                "ᚣ",
                "ᚪ",
                "ᚱ",
                "ᚾ",
                "ᛁ",
                "ᛖᛡ",
                "ᚪᛡ",
                "ᚪ",
                "ᚩᚢ",
                "ᚢ",
                "ᚩᛡ",
                "ᚩ",
                "ᚪᚱ",
                "ᚩ\u200dᚱ",
                "ᛖ\u200dᚱ",
                "ᛟᚱ",
                "ᛟᚱ",
                "ᛠᚱ",
                "ᛠ",
                "ᛄᚢ",
                "ᛁ",
            ],
        ][False - 1]
        cmap = {}
        if False == 3:
            cmap = {
                " ": "᛫",
                "-": "᛫",
                ",": "᛫᛫",
                ".": "᛬",
                "!": "᛭",
                ":": "᛬᛫",
                ";": "᛫᛬",
                "…": "᛫᛫᛫",
                "(": "[",
                ")": "]",
                "[": "(",
                "]": ")",
                "?": "?",
                "'": "",
                '"': "",
                "‘": "⟨",
                "’": "⟩",
                "“": "⟪",
                "”": "⟫",
            }
        inp = ">" + out + " "
        out = []
        squote = False
        dquote = False
        for i in range(len(inp) - 2):
            char = inp[i + 1]
            if char >= "𐑐" and char <= "𐒀":
                char = map[ord(char) - ord("𐑐")]
                if inp[i] == "·":
                    char = char[0].upper() + char[1:]
            if inp[i] == "<":
                content = False
            if inp[i] == ">":
                content = True
            if inp[i] in "𐑦𐑪𐑫𐑳":
                if char in "ᛈᛋᛒᛞᚸᛉᚷᚹᛗᚱᚾ":
                    char = char + "\u200d" + char
                elif char in "ᛏᚠᚦᚳᛚ":
                    char = char + char
                elif char == "ᛣ":
                    char = "ᛤ"
            if inp[i + 2] in "𐑤𐑮𐑘𐑿ᛄ" and char == "ᛣ":
                char = "ᚳ"
            if content and char in cmap:
                if char == "'":
                    char = "‘’"[squote]
                    squote = not squote
                if char == '"':
                    char = "“”"[dquote]
                    dquote = not dquote
                char = cmap[char] + "\u200b"
                if inp[i] in cmap and inp[i + 1] == " ":
                    char = ""
            if char != "·":
                out.append(char)
        out = "".join(out)
        for tup in [
            ("ᛁᛁ", "ᛡᛁ"),
            ("ᛁᛥᛄ", "ᛁᛋ\u200dᛋᚳ"),
            ("ᚾ\u200dᚾᛏ", "ᚾᚾᛏ"),
            ("ᚾᛏ", "ᚾ\u200dᛏ"),
            ("ᚣᛟᚱ", "ᚣᚱ"),
            ("ᚢᛟᚱ", "ᚢᚱ"),
            ("ᚪᛡᛟᚱ", "ᚪᛡᚱ"),
            ("ᛋᚳᛟᚾ", "ᛋᚳᚾ"),
            ("ᛋᚳᛄ", "ᛋᛣᛄ"),
            ("ᛉᚳᛟᚾ", "ᛉᚳᚾ"),
            ("ᛤᛥ", "ᛤᛋᛏ"),
        ]:
            out = out.replace(tup[0], tup[1])
    print(out, end="")
