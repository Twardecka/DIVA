REGISTRY = {}

from .basic_controller import BasicMAC
REGISTRY["basic_mac"] = BasicMAC
from .basic_controller_DIVA import BasicMACDIVA
REGISTRY["basic_mac_DIVA"] = BasicMACDIVA

from .casec_controller import CASECMAC
REGISTRY['casec_mac'] = CASECMAC
