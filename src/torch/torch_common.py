#!/usr/bin/env python3

import json
import os
import string
import time
import glob
import sys

import numpy as np

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from reserved_words import reserved

FEATURES = 77


def header(args, out):
    head_text = "# " + time.ctime(time.time())
    head_text += "\n# " + " ".join(args)
    for outfile in out:
        print(head_text, file=outfile)


def update_user(users, user):
    if user in reserved:
        return
    all_digit = True
    for char in user:
        if char not in string.digits:
            all_digit = False
    if all_digit:
        return
    users.add(user.lower())


def update_users(line, users):
    if len(line) < 2:
        return
    user = line[1]
    if user in [
        "Topic", "Signoff", "Signon", "Total", "#ubuntu"
        "Window", "Server:", "Screen:", "Geometry", "CO,",
        "Current", "Query", "Prompt:", "Second", "Split",
        "Logging", "Logfile", "Notification", "Hold", "Window",
        "Lastlog", "Notify", "netjoined:",
    ]:
        # Ignore as these are channel commands
        pass
    else:
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
        # This is for cases like a user named |blah| who is
        # refered to as simply blah
        core = [char for char in user]
        while len(core) > 0 and core[0] in string.punctuation:
            core.pop(0)
        while len(core) > 0 and core[-1] in string.punctuation:
            core.pop()
        core = "".join(core)
        update_user(users, core)


# Names two letters or less that occur more than 500 times in the data
common_short_names = {
    "ng", "_2", "x_", "rq", "\\9", "ww", "nn", "bc", "te",
    "io", "v7", "dm", "m0", "d1", "mr", "x3", "nm", "nu", "jc", "wy", "pa", "mn",
    "a_", "xz", "qr", "s1", "jo", "sw", "em", "jn", "cj", "j_",
}


def get_targets(line, users):
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


def lines_to_info(text_ascii):
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
        is_bot = (user == "ubottu" or user == "ubotu")
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

        info.append(
            (user, targets, chour, cmin, system, is_bot, last_from_user, line, next_from_user)
        )

    return info, target_info


def get_time_diff(info, a, b):
    if a is None or b is None:
        return -1
    if a > b:
        t = a
        a = b
        b = t
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


def get_features(name, query_no, link_no, text_ascii, text_tok, info, target_info, do_cache):
    global _cache
    if (name, query_no, link_no) in _cache:
        return _cache[name, query_no, link_no]

    features = []

    (
        quser, qtargets, qhour, qmin, qsystem, qis_bot, qlast_from_user, qline, qnext_from_user
    ) = info[query_no]
    (
        luser, ltargets, lhour, lmin, lsystem, lis_bot, llast_from_user, lline, lnext_from_user
    ) = info[link_no]

    # General information about this sample of data
    # Year
    for i in range(2004, 2018):
        features.append(str(i) in name)
    # Number of messages per minute
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
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= msg_per_min < end)

    # Query
    #  - Normal message or system message
    features.append(qsystem)
    #  - Hour of day
    features.append(qhour / 24)
    #  - Is it targeted
    features.append(len(qtargets) > 0)
    #  - Is there a previous message from this user?
    features.append(qlast_from_user is not None)
    #  - Did the previous message from this user have a target?
    if qlast_from_user is None:
        features.append(False)
    else:
        features.append(len(info[qlast_from_user][1]) > 0)
    #  - How long ago was the previous message from this user in messages?
    dist = -1 if qlast_from_user is None else query_no - qlast_from_user
    cutoffs = [-1, 0, 1, 5, 20, 1000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= dist < end)
    #  - How long ago was the previous message from this user in minutes?
    tdiff = get_time_diff(info, query_no, qlast_from_user)
    cutoffs = [-1, 0, 2, 10, 10000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= tdiff < end)
    #  - Are they a bot?
    features.append(qis_bot)

    # Link
    #  - Normal message or system message
    features.append(lsystem)
    #  - Hour of day
    features.append(lhour / 24)
    #  - Is it targeted
    features.append(link_no != query_no and len(ltargets) > 0)
    #  - Is there a previous message from this user?
    features.append(link_no != query_no and llast_from_user is not None)
    #  - Did the previous message from this user have a target?
    if link_no == query_no or llast_from_user is None:
        features.append(False)
    else:
        features.append(len(info[llast_from_user][1]) > 0)
    #  - How long ago was the previous message from this user in messages?
    dist = -1 if llast_from_user is None else link_no - llast_from_user
    cutoffs = [-1, 0, 1, 5, 20, 1000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(link_no != query_no and start <= dist < end)
    #  - How long ago was the previous message from this user in minutes?
    tdiff = get_time_diff(info, link_no, llast_from_user)
    cutoffs = [-1, 0, 2, 10, 10000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= tdiff < end)
    #  - Are they a bot?
    features.append(lis_bot)
    #  - Is the message after from the same user?
    features.append(
        link_no != query_no and link_no + 1 < len(info) and luser == info[link_no + 1][0]
    )
    #  - Is the message before from the same user?
    features.append(
        link_no != query_no and link_no - 1 > 0 and luser == info[link_no - 1][0]
    )

    # Both
    #  - Is this a self-link?
    features.append(link_no == query_no)
    #  - How far apart in messages are the two?
    dist = query_no - link_no
    features.append(min(100, dist) / 100)
    features.append(dist > 1)
    #  - How far apart in time are the two?
    tdiff = get_time_diff(info, link_no, query_no)
    features.append(min(100, tdiff) / 100)
    cutoffs = [-1, 0, 1, 5, 60, 10000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= tdiff < end)
    #  - Does the link target the query user?
    features.append(quser.lower() in ltargets)
    #  - Does the query target the link user?
    features.append(luser.lower() in qtargets)
    #  - none in between from src?
    features.append(link_no != query_no and (qlast_from_user is None or qlast_from_user < link_no))
    #  - none in between from target?
    features.append(link_no != query_no and (lnext_from_user is None or lnext_from_user > query_no))
    #  - previously src addressed target?
    #  - future src addressed target?
    #  - src addressed target in between?
    if link_no != query_no and (quser, luser) in target_info:
        features.append(min(target_info[quser, luser]) < link_no)
        features.append(max(target_info[quser, luser]) > query_no)
        between = False
        for num in target_info[quser, luser]:
            if query_no > num > link_no:
                between = True
        features.append(between)
    else:
        features.append(False)
        features.append(False)
        features.append(False)
    #  - previously target addressed src?
    #  - future target addressed src?
    #  - target addressed src in between?
    if link_no != query_no and (luser, quser) in target_info:
        features.append(min(target_info[luser, quser]) < link_no)
        features.append(max(target_info[luser, quser]) > query_no)
        between = False
        for num in target_info[luser, quser]:
            if query_no > num > link_no:
                between = True
        features.append(between)
    else:
        features.append(False)
        features.append(False)
        features.append(False)
    #  - are they the same speaker?
    features.append(luser == quser)
    #  - do they have the same target?
    features.append(link_no != query_no and len(ltargets.intersection(qtargets)) > 0)
    #  - Do they have words in common?
    ltokens = set(text_ascii[link_no])
    qtokens = set(text_ascii[query_no])
    common = len(ltokens.intersection(qtokens))
    if link_no != query_no and len(ltokens) > 0 and len(qtokens) > 0:
        features.append(common / len(ltokens))
        features.append(common / len(qtokens))
    else:
        features.append(False)
        features.append(False)
    features.append(link_no != query_no and common == 0)
    features.append(link_no != query_no and common == 1)
    features.append(link_no != query_no and common > 1)
    features.append(link_no != query_no and common > 5)

    # Convert to 0/1
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


def load_conversations(filenames, is_test, test_start, test_end):
    conversations = []
    done = set()
    for filename in expand_input_filenames(filenames):
        name = filename
        for ending in [".annotation.txt", ".ascii.txt", ".raw.txt", ".tok.txt"]:
            if filename.endswith(ending):
                name = filename[: -len(ending)]
        if name in done:
            continue
        done.add(name)
        text_ascii = [l.strip().split() for l in open(name + ".ascii.txt")]
        text_tok = []
        for l in open(name + ".tok.txt"):
            l = l.strip().split()
            if len(l) > 0 and l[-1] == "</s>":
                l = l[:-1]
            if len(l) == 0 or l[0] != "<s>":
                l.insert(0, "<s>")
            text_tok.append(l)
        info, target_info = lines_to_info(text_ascii)

        links = {}
        if is_test:
            for i in range(test_start, min(test_end, len(text_ascii))):
                links[i] = []
        else:
            for line in open(name + ".annotation.txt"):
                nums = [int(v) for v in line.strip().split() if v != "-"]
                links.setdefault(max(nums), []).append(min(nums))
        conversations.append(
            {
                "name": name + ".annotation.txt",
                "text_ascii": text_ascii,
                "text_tok": text_tok,
                "info": info,
                "target_info": target_info,
                "links": links,
            }
        )
    return conversations


def load_embeddings(path):
    id_to_token = []
    token_to_id = {}
    pretrained = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) <= 2:
                continue
            word = parts[0].lower()
            vector = [float(v) for v in parts[1:]]
            token_to_id[word] = len(id_to_token)
            id_to_token.append(word)
            pretrained.append(vector)
    if len(pretrained) == 0:
        raise ValueError("No embeddings found in {}".format(path))
    weights = np.array(pretrained, dtype=np.float32)
    return token_to_id, id_to_token, weights


def get_ids(words, token_to_id):
    ans = []
    backup = token_to_id.get("<unka>", 0)
    for word in words:
        ans.append(token_to_id.get(word, backup))
    return ans


def write_manifest(manifest_path, records):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def expand_input_filenames(filenames):
    expanded = []
    for filename in filenames:
        if glob.has_magic(filename):
            matches = sorted(glob.glob(filename))
            if len(matches) == 0:
                raise FileNotFoundError("No files matched pattern: {}".format(filename))
            expanded.extend(matches)
        else:
            expanded.append(filename)
    return expanded

