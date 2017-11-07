"""
@file    roundabout.py
@author  Daniel Krajzewicz
@author  Laura Bieker
@author  Karol Stosiek
@author  Michael Behrisch
@date    2011-11-28
@version $Id: roundabout.py 22608 2017-01-17 06:28:54Z behrisch $

This file contains a Python-representation of a single roundabout.

SUMO, Simulation of Urban MObility; see http://sumo.dlr.de/
Copyright (C) 2008-2017 DLR (http://www.dlr.de/) and contributors

This file is part of SUMO.
SUMO is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.
"""


class Roundabout:

    def __init__(self, nodes, edges=None):
        self._nodes = nodes
        self._edges = edges

    def getNodes(self):
        return self._nodes

    def getEdges(self):
        return self._edges
