import random

choices = ['1v1g','1v2g','1v3g','1v4g','1v5g','1v6g','2v1g','2v2g','2v3g','2v4g','2v5g','2v6g','3v1g','3v2g','3v3g','3v4g','3v5g','3v6g','4v1g','4v2g','4v3g','4v4g','4v5g','4v6g','5v1g','5v2g','5v3g','5v4g','5v5g','5v6g','6v1g','6v2g','6v3g','6v4g','6v5g','6v6g'];

random.shuffle(choices)

for i in range(10):
	runtime = choices.pop()
	# print in the format of spreadsheet
	print runtime[-2:] + " " + runtime[:2]
