"""
@file    poi.py
@author  Daniel Krajzewicz
@author  Michael Behrisch
@author  Jakob Erdmann
@author  Melanie Knocke
@date    2010-02-18
@version $Id: poi.py 23247 2017-03-07 13:46:58Z behrisch $

Library for reading and storing PoIs.

SUMO, Simulation of Urban MObility; see http://sumo.dlr.de/
Copyright (C) 2010-2017 DLR (http://www.dlr.de/) and contributors

This file is part of SUMO.
SUMO is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.
"""
from __future__ import absolute_import

from xml.sax import handler, parse
from .. import color


class PoI:

    def __init__(self, id, type, layer, color, x, y, lane=None, pos=None, lonLat=False):
        """interpret x,y as lon,lat if lonLat is True"""
        self.id = id
        self.type = type
        self.color = color
        self.layer = layer
        self.x = x
        self.y = y
        self.lane = lane
        self.pos = pos
        self.attributes = {}
        self.lonLat = lonLat

    def toXML(self):
        if self.lane:
            ret = '<poi id="%s" type="%s" color="%s" layer="%s" lane="%s" pos="%s"' % (
                self.id, self.type, self.color.toXML(), self.layer, self.lane, self.pos)
        elif self.lonLat:
            ret = '<poi id="%s" type="%s" color="%s" layer="%s" lon="%s" lat="%s"' % (
                self.id, self.type, self.color.toXML(), self.layer, self.x, self.y)
        else:
            ret = '<poi id="%s" type="%s" color="%s" layer="%s" x="%s" y="%s"' % (
                self.id, self.type, self.color.toXML(), self.layer, self.x, self.y)
        if len(self.attributes) == 0:
            ret += '/>'
        else:
            ret += '>'
            for a in self.attributes:
                ret += '<param key="%s" value="%s"/>' % (a, self.attributes[a])
            ret += '</poi>'
        return ret


class PoIReader(handler.ContentHandler):

    def __init__(self):
        self._id2poi = {}
        self._pois = []
        self._lastPOI = None

    def startElement(self, name, attrs):
        if name == 'poi':
            c = color.decodeXML(attrs['color'])
            if 'lane' not in attrs:
                if 'x' not in attrs:
                    poi = PoI(attrs['id'], attrs['type'], float(attrs['layer']), c, float(
                        attrs['lon']), float(attrs['lat']), lonLat=True)
                else:
                    poi = PoI(attrs['id'], attrs['type'], float(
                        attrs['layer']), c, float(attrs['x']), float(attrs['y']))
            else:
                poi = PoI(attrs['id'], attrs['type'], float(
                    attrs['layer']), c, None, None, attrs['lane'], float(attrs['pos']))
            self._id2poi[poi.id] = poi
            self._pois.append(poi)
            self._lastPOI = poi
        if name == 'param' and self._lastPOI is not None:
            self._lastPOI.attributes[attrs['key']] = attrs['value']

    def endElement(self, name):
        if name == 'poi':
            self._lastPOI = None


def read(filename):
    pois = PoIReader()
    parse(filename, pois)
    return pois._pois
