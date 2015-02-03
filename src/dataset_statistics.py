#!/usr/bin/python
# -*- coding: utf-8 -*-

import utils
from db_layer import db_layer


def main():
    db = db_layer("conf/config.json")
    languages = db.get_languages()
    authors = db.get_authors()
    print "total", len(authors)
    for ln in languages:
        authors = db.get_authors(ln)
        print ln, len(authors)
        print "authors", authors[:10], "..."
        
        db.get_author(authors[0])

    print "Bye!"


if __name__ == "__main__":
    main()
