#!/usr/bin/env python
import os
from dotenv import load_dotenv
load_dotenv()

BASILICA_KEY = os.getenv('BASILICA_KEY')

TITLE = os.getenv('TITLE')
SUBTITLE = os.getenv('SUBTITLE')

PORT = os.getenv('PORT')
