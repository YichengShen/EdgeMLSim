# Definition of a message class

import pickle
from enum import Enum

class InstanceType(Enum):
	CLOUD_SERVER = 0
	EDGE_SERVER = 1
	WORKER = 2
	SIMULATOR = 9

class PayloadType(Enum):
	CONNECTION_SIGNAL = 0
	MODEL = 1
	PARAMETER = 2
	GRADIENT = 3
	START_MESSAGE = 4
	DATA = 5
	ID = 6
	REQUEST = 7

class Msg:
	def __init__(self, source_type, payload_type, payload):
		self.source_type = source_type
		self.payload_type = payload_type
		self.payload = payload

	def serialize(self):
		return pickle.dumps(self)

	def get_source_type(self):
		return self.source_type

	def get_payload_type(self):
		return self.payload_type

	def get_payload(self):
		return self.payload