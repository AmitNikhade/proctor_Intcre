# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = "ACaba66f2dc52fbdf50d36f0266100dee3"
auth_token = "71f3be49bee855c1349778c4deb32c2b"
client = Client(account_sid, auth_token)

token = client.tokens.create()

print(token.ice_servers)

# [{'url': 'stun:global.stun.twilio.com:3478', 'urls': 'stun:global.stun.twilio.com:3478'}, {'url': 'turn:global.turn.twilio.com:3478?transport=udp', 'username': '8e91ff2ef4eff00645d3776d8e18e057f6417d8ce04584591f0d950e8bd16f97', 'urls': 'turn:global.turn.twilio.com:3478?transport=udp', 'credential': '9K6+45gg6mjEpOt5AKFHZ6qHK2Iuzuj5nH0GoqRJmSc='}, {'url': 'turn:global.turn.twilio.com:3478?transpor[{'url': 'stun:global.stun.twilio.com:3478', 'urls': 'stun:global.stun.twilio.com:3478'}, {'url': 'turn:global.turn.twilio.com:3478?transport=udp', 'username': '8e91ff2ef4eff00645d3776d8e18e057f6417d8ce04584591f0d950e8bd16f97', 'urls': 'turn:global.turn.twilio.com:3478?transport=udp', 'credential': '9K6+45gg6mjEpOt5AKFHZ6qHK2Iuzuj5nH0GoqRJmSc='}, {'url': 'turn:global.turn.twilio.com:3478?transpor478?transport=udp', 'credential': '9K6+45gg6mjEpOt5AKFHZ6qHK2Iuzuj5nH0GoqRJmSc='}, {'url': 'turn:global.turn.twilio.com:3478?transport=tcp', 'username': '8e91ff2ef4eff00645d3776d8e18e057f6417d8ce04584591f0d950e8bd16f97', 'urls': 'turn:global.turn.twilio.com:3478?transport=tcp', 'credential': '9K6+45gg6mjEpOt5AKFHZ6qHK2Iuzuj5nH0GoqRJmSc='}, {'url': 'turn:global.turn.twilio.com:443?transport=tcp', 'username': '8e91ff2ef4eff00645d3776d8e18e057f6417d8ce04584591f0d950e8bd16f97', 'urls': 'turn:global.turn.twilio.com:443?transport=tcp', 'credential': '9K6+45gg6mjEpOt5AKFHZ6qHK2Iuzuj5nH0GoqRJmSc='}]