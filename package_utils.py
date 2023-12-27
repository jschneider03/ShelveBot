import os
from manipulation import ConfigureParser
from settings import PACKAGE_XML_PATH


def CustomConfigureParser(parser):
    parser.package_map().AddPackageXml(PACKAGE_XML_PATH)
    ConfigureParser(parser)
    return parser


def GetPathFromUrl(url: str, parser):
    assert url.startswith('package://')
    url = url.removeprefix('package://')
    terms = url.split('/')
    package_name = terms[0]
    rest = "/".join(terms[1:])
    return os.path.join(parser.package_map().GetPath(package_name), rest)


def GetModelNameFromUrl(url: str):
    assert url.startswith('package://')
    url = url.removeprefix('package://')
    terms = url.split('/')
    return terms[-1]
