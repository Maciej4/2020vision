import socket

def server():
	host = socket.gethostname()
	port = 8080
	
	s. = socket.socket()
	s.bind((host,port))
	
	s.listen(1)
	client_socket, adress = s.accept()
	print("Connection from: " + str(addr))
	while True:
		data = c.recv(1024).decode('utf-8')
		if not data:
			break
		print('From online user: ' + data)
		data = data.upper()
		c.send(data.encode('utf-8'))
	c.close()

if __name__ = '__main__'
	server()
