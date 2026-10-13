#!/usr/bin/env python3
"""Produce the revised manuscript as a real Word tracked-changes document.

Deletions are emitted as <w:del>/<w:delText> and insertions as <w:ins>/<w:t>, so
Word's Review pane can accept or reject each change individually. Both are also
coloured red — deletions struck through, insertions underlined — so the revision
is legible even with mark-up display turned off or in a PDF export.
"""
from __future__ import annotations

import copy
import re
from datetime import datetime, timezone

from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

AUTHOR = "Revision"
DATE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
RED = "C00000"
_counter = [1000]


def _nid():
    _counter[0] += 1
    return str(_counter[0])


def _rpr(strike=False, underline=False, base=None):
    rpr = copy.deepcopy(base) if base is not None else OxmlElement("w:rPr")
    for tag in ("w:color", "w:strike", "w:u"):
        for el in rpr.findall(qn(tag)):
            rpr.remove(el)
    color = OxmlElement("w:color")
    color.set(qn("w:val"), RED)
    rpr.append(color)
    if strike:
        rpr.append(OxmlElement("w:strike"))
    if underline:
        u = OxmlElement("w:u")
        u.set(qn("w:val"), "single")
        rpr.append(u)
    return rpr


def _run(text, deleted=False, base_rpr=None):
    r = OxmlElement("w:r")
    r.append(_rpr(strike=deleted, underline=not deleted, base=base_rpr))
    t = OxmlElement("w:delText" if deleted else "w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    r.append(t)
    return r


def _wrap(tag, child):
    el = OxmlElement(tag)
    el.set(qn("w:id"), _nid())
    el.set(qn("w:author"), AUTHOR)
    el.set(qn("w:date"), DATE)
    el.append(child)
    return el


def track_replace(par, new_text):
    """Mark every run in `par` deleted and append `new_text` as an insertion."""
    p = par._p
    base_rpr = None
    old_runs = p.findall(qn("w:r"))
    if old_runs:
        base_rpr = old_runs[0].find(qn("w:rPr"))
    old_text = "".join(r.text for r in par.runs)
    for r in old_runs:
        p.remove(r)
    if old_text.strip():
        p.append(_wrap("w:del", _run(old_text, deleted=True, base_rpr=base_rpr)))
    if new_text.strip():
        p.append(_wrap("w:ins", _run(new_text, deleted=False, base_rpr=base_rpr)))
    return par


def track_delete(par):
    return track_replace(par, "")


def track_insert_after(par, text, style=None):
    """Insert a wholly new paragraph after `par`, marked as an insertion."""
    newp = copy.deepcopy(par._p)
    for child in list(newp):
        if child.tag in (qn("w:r"), qn("w:ins"), qn("w:del")):
            newp.remove(child)
    par._p.addnext(newp)
    from docx.text.paragraph import Paragraph
    np = Paragraph(newp, par._parent)
    if style is not None:
        try:
            np.style = style
        except KeyError:
            pass
    # mark the paragraph mark itself as inserted so Word can reject the whole line
    pPr = newp.find(qn("w:pPr"))
    if pPr is None:
        pPr = OxmlElement("w:pPr")
        newp.insert(0, pPr)
    rPr = pPr.find(qn("w:rPr"))
    if rPr is None:
        rPr = OxmlElement("w:rPr")
        pPr.append(rPr)
    ins = OxmlElement("w:ins")
    ins.set(qn("w:id"), _nid())
    ins.set(qn("w:author"), AUTHOR)
    ins.set(qn("w:date"), DATE)
    rPr.insert(0, ins)
    newp.append(_wrap("w:ins", _run(text)))
    return np


def add_comment_note(par, text):
    """A visible margin-style note, as an inserted italic red line."""
    np = track_insert_after(par, "[Revision note] " + text)
    for r in np._p.iter(qn("w:rPr")):
        r.append(OxmlElement("w:i"))
    return np


def find(doc, needle, start=0):
    for i in range(start, len(doc.paragraphs)):
        if needle.lower() in doc.paragraphs[i].text.lower():
            return i
    raise LookupError(needle)


def enable_track_changes(doc):
    settings = doc.settings.element
    for tag in ("w:trackChanges",):
        if settings.find(qn(tag)) is None:
            settings.append(OxmlElement(tag))
