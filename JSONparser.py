import json

strJSON = open("dumpJSON.txt", "r").read()

JSON = json.loads(strJSON)

f = open("parsedJSON.txt","w+")

for i in range(len(JSON)):
    f.write("%d,%s,%d,%d\n" % (JSON[i]["id"], "isImportant", JSON[i]["x"], JSON[i]["y"]))


