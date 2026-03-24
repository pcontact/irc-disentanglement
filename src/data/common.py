#!/usr/bin/env python3

import hashlib
import json
import os
import string
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from reserved_words import reserved

FEATURES = 77
SCHEMA_VERSION = "dynet-precompute-v1"


def header(args: Sequence[str], out) -> None:
    head_text = "# " + time.ctime(time.time())
    head_text += "\n# " + " ".join(args)
    for outfile in out:
        print(head_text, file=outfile)


def get_log_path(prefix: str, suffix: str) -> str:
    return prefix + suffix


def resolve_name(filename: str) -> str:
    name = filename
    for ending in [".annotation.txt", ".ascii.txt", ".raw.txt", ".tok.txt"]:
        if filename.endswith(ending):
            name = filename[: -len(ending)]
    return name


def safe_name(name: str) -> str:
    return name.replace("\\", "_").replace("/", "_").replace(":", "")


def update_user(users: set, user: str) -> None:
    if user in reserved:
        return
    all_digit = True
    for char in user:
        if char not in string.digits:
            all_digit = False
    if all_digit:
        return
    users.add(user.lower())


def update_users(line: Sequence[str], users: set) -> None:
    if len(line) < 2:
        return
    user = line[1]
    if user in [
        "Topic",
        "Signoff",
        "Signon",
        "Total",
        "#ubuntu" "Window",
        "Server:",
        "Screen:",
        "Geometry",
        "CO,",
        "Current",
        "Query",
        "Prompt:",
        "Second",
        "Split",
        "Logging",
        "Logfile",
        "Notification",
        "Hold",
        "Window",
        "Lastlog",
        "Notify",
        "netjoined:",
    ]:
        return

    if line[0].endswith("==="):
        parts = " ".join(line).split("is now known as")
        if len(parts) == 2 and line[-1] == parts[-1].strip():
            user = line[-1]
    elif line[0][-1] == "]":
        if user[0] == "<":
            user = user[1:]
        if user[-1] == ">":
            user = user[:-1]

    user = user.lower()
    update_user(users, user)
    core = [char for char in user]
    while len(core) > 0 and core[0] in string.punctuation:
        core.pop(0)
    while len(core) > 0 and core[-1] in string.punctuation:
        core.pop()
    update_user(users, "".join(core))


common_short_names = {
    "ng",
    "_2",
    "x_",
    "rq",
    "\\9",
    "ww",
    "nn",
    "bc",
    "te",
    "io",
    "v7",
    "dm",
    "m0",
    "d1",
    "mr",
    "x3",
    "nm",
    "nu",
    "jc",
    "wy",
    "pa",
    "mn",
    "a_",
    "xz",
    "qr",
    "s1",
    "jo",
    "sw",
    "em",
    "jn",
    "cj",
    "j_",
}


def get_targets(line: Sequence[str], users: set) -> set:
    targets = set()
    for token in line[2:]:
        token = token.lower()
        user = None
        if token in users and len(token) > 2:
            user = token
        else:
            core = [char for char in token]
            while len(core) > 0 and core[-1] in string.punctuation:
                core.pop()
                nword = "".join(core)
                if nword in users and (len(core) > 2 or nword in common_short_names):
                    user = nword
                    break
            if user is None:
                while len(core) > 0 and core[0] in string.punctuation:
                    core.pop(0)
                    nword = "".join(core)
                    if nword in users and (len(core) > 2 or nword in common_short_names):
                        user = nword
                        break
        if user is not None:
            targets.add(user)
    return targets


def lines_to_info(text_ascii: Sequence[Sequence[str]]):
    users = set()
    for line in text_ascii:
        update_users(line, users)

    chour = 12
    cmin = 0
    info = []
    target_info = {}
    nexts = {}
    for line_no, line in enumerate(text_ascii):
        if line[0].startswith("["):
            user = line[1][1:-1]
            nexts.setdefault(user, []).append(line_no)

    prev = {}
    for line_no, line in enumerate(text_ascii):
        user = line[1]
        system = True
        if line[0].startswith("["):
            chour = int(line[0][1:3])
            cmin = int(line[0][4:6])
            user = user[1:-1]
            system = False
        is_bot = user in {"ubottu", "ubotu"}
        targets = get_targets(line, users)
        for target in targets:
            target_info.setdefault((user, target), []).append(line_no)
        last_from_user = prev.get(user, None)
        if not system:
            prev[user] = line_no
        next_from_user = None
        if user in nexts:
            while len(nexts[user]) > 0 and nexts[user][0] <= line_no:
                nexts[user].pop(0)
            if len(nexts[user]) > 0:
                next_from_user = nexts[user][0]

        info.append((user, targets, chour, cmin, system, is_bot, last_from_user, line, next_from_user))

    return info, target_info


def get_time_diff(info, a: Optional[int], b: Optional[int]) -> int:
    if a is None or b is None:
        return -1
    if a > b:
        a, b = b, a
    ahour = info[a][2]
    amin = info[a][3]
    bhour = info[b][2]
    bmin = info[b][3]
    if ahour == bhour:
        return bmin - amin
    if bhour < ahour:
        bhour += 24
    return (60 - amin) + bmin + 60 * (bhour - ahour - 1)


_cache = {}


def get_features(name, query_no, link_no, text_ascii, info, target_info, do_cache=True):
    global _cache
    if do_cache and (name, query_no, link_no) in _cache:
        return _cache[name, query_no, link_no]

    features = []
    (
        quser,
        qtargets,
        qhour,
        _qmin,
        qsystem,
        qis_bot,
        qlast_from_user,
        _qline,
        _qnext_from_user,
    ) = info[query_no]
    (
        luser,
        ltargets,
        lhour,
        _lmin,
        lsystem,
        lis_bot,
        llast_from_user,
        _lline,
        lnext_from_user,
    ) = info[link_no]

    for year in range(2004, 2018):
        features.append(str(year) in name)

    start = None
    end = None
    for i in range(len(text_ascii)):
        if start is None and text_ascii[i][0].startswith("["):
            start = i
        if end is None and i > 0 and text_ascii[-i][0].startswith("["):
            end = len(text_ascii) - i - 1
        if start is not None and end is not None:
            break
    diff = get_time_diff(info, start, end)
    msg_per_min = len(text_ascii) / max(1, diff)
    cutoffs = [-1, 1, 3, 10, 10000]
    for low, high in zip(cutoffs, cutoffs[1:]):
        features.append(low <= msg_per_min < high)

    features.append(qsystem)
    features.append(qhour / 24)
    features.append(len(qtargets) > 0)
    features.append(qlast_from_user is not None)
    if qlast_from_user is None:
        features.append(False)
    else:
        features.append(len(info[qlast_from_user][1]) > 0)
    dist = -1 if qlast_from_user is None else query_no - qlast_from_user
    cutoffs = [-1, 0, 1, 5, 20, 1000]
    for low, high in zip(cutoffs, cutoffs[1:]):
        features.append(low <= dist < high)
    time_diff = get_time_diff(info, query_no, qlast_from_user)
    cutoffs = [-1, 0, 2, 10, 10000]
    for low, high in zip(cutoffs, cutoffs[1:]):
        features.append(low <= time_diff < high)
    features.append(qis_bot)

    features.append(lsystem)
    features.append(lhour / 24)
    features.append(link_no != query_no and len(ltargets) > 0)
    features.append(link_no != query_no and llast_from_user is not None)
    if link_no == query_no or llast_from_user is None:
        features.append(False)
    else:
        features.append(len(info[llast_from_user][1]) > 0)
    dist = -1 if llast_from_user is None else link_no - llast_from_user
    cutoffs = [-1, 0, 1, 5, 20, 1000]
    for low, high in zip(cutoffs, cutoffs[1:]):
        features.append(link_no != query_no and low <= dist < high)
    time_diff = get_time_diff(info, link_no, llast_from_user)
    cutoffs = [-1, 0, 2, 10, 10000]
    for low, high in zip(cutoffs, cutoffs[1:]):
        features.append(low <= time_diff < high)
    features.append(lis_bot)
    features.append(link_no != query_no and link_no + 1 < len(info) and luser == info[link_no + 1][0])
    features.append(link_no != query_no and link_no - 1 > 0 and luser == info[link_no - 1][0])

    features.append(link_no == query_no)
    dist = query_no - link_no
    features.append(min(100, dist) / 100)
    features.append(dist > 1)
    time_diff = get_time_diff(info, link_no, query_no)
    features.append(min(100, time_diff) / 100)
    cutoffs = [-1, 0, 1, 5, 60, 10000]
    for low, high in zip(cutoffs, cutoffs[1:]):
        features.append(low <= time_diff < high)
    features.append(quser.lower() in ltargets)
    features.append(luser.lower() in qtargets)
    features.append(link_no != query_no and (qlast_from_user is None or qlast_from_user < link_no))
    features.append(link_no != query_no and (lnext_from_user is None or lnext_from_user > query_no))

    if link_no != query_no and (quser, luser) in target_info:
        features.append(min(target_info[quser, luser]) < link_no)
        features.append(max(target_info[quser, luser]) > query_no)
        between = False
        for num in target_info[quser, luser]:
            if query_no > num > link_no:
                between = True
        features.append(between)
    else:
        features.extend([False, False, False])

    if link_no != query_no and (luser, quser) in target_info:
        features.append(min(target_info[luser, quser]) < link_no)
        features.append(max(target_info[luser, quser]) > query_no)
        between = False
        for num in target_info[luser, quser]:
            if query_no > num > link_no:
                between = True
        features.append(between)
    else:
        features.extend([False, False, False])

    features.append(luser == quser)
    features.append(link_no != query_no and len(ltargets.intersection(qtargets)) > 0)
    ltokens = set(text_ascii[link_no])
    qtokens = set(text_ascii[query_no])
    common = len(ltokens.intersection(qtokens))
    if link_no != query_no and len(ltokens) > 0 and len(qtokens) > 0:
        features.append(common / len(ltokens))
        features.append(common / len(qtokens))
    else:
        features.extend([False, False])
    features.append(link_no != query_no and common == 0)
    features.append(link_no != query_no and common == 1)
    features.append(link_no != query_no and common > 1)
    features.append(link_no != query_no and common > 5)

    final_features = []
    for feature in features:
        if feature is True:
            final_features.append(1.0)
        elif feature is False:
            final_features.append(0.0)
        else:
            final_features.append(feature)

    if do_cache:
        _cache[name, query_no, link_no] = final_features
    return final_features


def clear_feature_cache() -> None:
    _cache.clear()


def load_tokenised_text(name: str) -> List[List[str]]:
    text_tok = []
    with open(name + ".tok.txt", "r", encoding="utf-8") as handle:
        for line in handle:
            tokens = line.strip().split()
            if len(tokens) > 0 and tokens[-1] == "</s>":
                tokens = tokens[:-1]
            if len(tokens) == 0 or tokens[0] != "<s>":
                tokens.insert(0, "<s>")
            text_tok.append(tokens)
    return text_tok


def load_ascii_text(name: str) -> List[List[str]]:
    with open(name + ".ascii.txt", "r", encoding="utf-8") as handle:
        return [line.strip().split() for line in handle]


def load_links(name: str, is_test: bool, test_start: int, test_end: int, message_count: int):
    links = {}
    if is_test:
        for i in range(test_start, min(test_end, message_count)):
            links[i] = []
    else:
        with open(name + ".annotation.txt", "r", encoding="utf-8") as handle:
            for line in handle:
                nums = [int(v) for v in line.strip().split() if v != "-"]
                links.setdefault(max(nums), []).append(min(nums))
    return links


def load_conversations(filenames, is_test=False, test_start=1000, test_end=1000000):
    conversations = []
    done = set()
    for filename in filenames or []:
        name = resolve_name(filename)
        if name in done:
            continue
        done.add(name)
        text_ascii = load_ascii_text(name)
        text_tok = load_tokenised_text(name)
        info, target_info = lines_to_info(text_ascii)
        links = load_links(name, is_test, test_start, test_end, len(text_ascii))
        conversations.append(
            {
                "name": name,
                "text_ascii": text_ascii,
                "text_tok": text_tok,
                "info": info,
                "target_info": target_info,
                "links": links,
            }
        )
    return conversations


def compute_embedding_hash(word_vectors: Optional[str]) -> Optional[str]:
    if not word_vectors:
        return None
    digest = hashlib.sha256()
    dim = None
    with open(word_vectors, "rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if dim is None:
                parts = raw_line.decode("utf-8").strip().split()
                dim = max(0, len(parts) - 1)
    digest.update(("dim:{}".format(dim)).encode("utf-8"))
    return digest.hexdigest()


def load_embeddings(word_vectors: str):
    token_to_id = {}
    id_to_token = []
    pretrained = []
    embedding_hash = compute_embedding_hash(word_vectors)
    with open(word_vectors, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            word = parts[0].lower()
            vector = [float(v) for v in parts[1:]]
            token_to_id[word] = len(id_to_token)
            id_to_token.append(word)
            pretrained.append(vector)
    weights = np.array(pretrained, dtype=np.float32) if pretrained else None
    return token_to_id, id_to_token, weights, embedding_hash


def get_ids(words: Sequence[str], token_to_id: Optional[Dict[str, int]]) -> List[int]:
    if token_to_id is None:
        return []
    backup = token_to_id.get("<unka>", 0)
    return [token_to_id.get(word, backup) for word in words]


def write_manifest(manifest_path: str, manifest_records: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for record in manifest_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
