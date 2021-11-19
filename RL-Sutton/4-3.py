import csv
import json
import math

csv_file = open('1.csv', 'r')
keys = csv_file.readline().strip('\n').split(',')

jsonInfo = []

while True:
    values = csv_file.readline().strip('\n').split(',')
    if values == ['']:
        break
    dic_temp = dict(zip(keys,values)) 
    jsonInfo.append(dic_temp)

csv_file.close()

elem_list = {}

for key in keys:
    temp_tuple = set()
    for info in jsonInfo:
        temp_tuple.add(info.get(key))
    elem_list[key] = temp_tuple

def checkType(examples):
    lastType = jsonInfo[examples[0]].get('isGood')
    for i in examples:
        if jsonInfo[i].get('isGood') != lastType:
            return False
    return True

def checkAttrIsNull(attributes):
    if len(attributes) == 0:
        return True
    return (False)

def isAttributeSame(attributes, examples):
    for attr in attributes:
        lastType = jsonInfo[examples[0]].get(attr)
        for i in examples:
            if jsonInfo[i].get(attr) != lastType:
                return False
    return True

def getMostType(examples):
    typeTrue = 0
    typeFalse = 0
    for i in examples:
        if jsonInfo[i].get('isGood') == '0':
            typeFalse += 1
        else:
            typeTrue += 1
    return typeTrue >= typeFalse

def calculateEnt(examples):
    if len(examples) == 0:
        return 0
    typeTrue = 0
    typeFalse = 0
    res = 0
    for i in examples:
        if int(jsonInfo[i].get('isGood')) == 0:
            typeFalse += 1
        else:
            typeTrue += 1
    typeFalse = typeFalse / len(examples)
    typeTrue = typeTrue / len(examples)
    if typeFalse != 0:
        res -= (typeFalse * math.log2(typeFalse))
    if typeTrue != 0:
        res -= (typeTrue * math.log2(typeTrue))
    return res

def calculateGain(examples, attrs):
    res = 0
    resMargin = 0
    if attrs != 'density' and attrs != 'sugar':
        res += calculateEnt(examples)
        for attr in elem_list[attrs]:
            examples_son = []
            for i in examples:
                if jsonInfo[i].get(attrs) == attr:
                    examples_son.append(i)
            res -= (len(examples_son) / len(examples)) * calculateEnt(examples_son)
        return [res,0.0]
    else:
        margins = []
        margins2 = []
        for x in elem_list[attrs]:
            margins.append(float(x))
        margins = sorted(margins)
        for i in range(len(margins) - 1):
            margins2.append((margins[i]+margins[i+1])/2)
        for x in margins2:
            res2 = 0
            res2 += calculateEnt(examples)
            examples_son = []
            examples_son2 = []
            for i in examples:
                if float(jsonInfo[i].get(attrs)) < x:
                    examples_son.append(i)
                elif float(jsonInfo[i].get(attrs)) > x:
                    examples_son2.append(i)
            res2 -= (len(examples_son) / len(examples)) * calculateEnt(examples_son)
            res2 -= (len(examples_son2) / len(examples)) * calculateEnt(examples_son2)
            if res2 > res:
                resMargin = x
                res = res2
        return [res,resMargin]

nodeRelation = {}

def TreeGenerate(examples, attributes, fatherNodeId, nodeId, elem):
    nodeId += 1
    nownode = nodeId
    nodeRelation[str(nodeId)] = {}
    nodeRelation[str(nodeId)]['father'] = fatherNodeId
    nodeRelation[str(nodeId)]['attrValue'] = elem
    if checkType(examples):
        nodeRelation[str(nodeId)]['isGood'] = jsonInfo[examples[0]].get('isGood')
        return nownode
    elif checkAttrIsNull(attributes):
        nodeRelation[str(nodeId)]['isGood'] = getMostType(examples)
        return nownode
    elif isAttributeSame(attributes, examples):
        nodeRelation[str(nodeId)]['isGood'] = getMostType(examples)
        return nownode
    
    maxGain = 0
    maxAttr = ""
    maxMargin = 0.0
    for attr in attributes:
        Gain = calculateGain(examples, attr)
        if maxGain < Gain[0]:
            maxGain = Gain[0]
            maxAttr = attr
            maxMargin = Gain[1]
    nodeRelation[str(nodeId)]['attr'] = maxAttr
    if maxAttr != 'density' and maxAttr != 'sugar':
        for elem in elem_list[maxAttr]:
            examples_son = []
            for i in examples:
                if jsonInfo[i].get(maxAttr) == elem:
                    examples_son.append(i)
            if len(examples_son) == 0:
                nodeRelation[str(nodeId)]['isGood'] = getMostType(examples)
                return nownode
            else:
                nextAttrs = []
                for x in attributes:
                    if x == maxAttr:
                        continue
                    nextAttrs.append(x)
                nodeId = TreeGenerate(examples_son, nextAttrs, nownode, nodeId, elem)
    else:
        examples_son = []
        examples_son2 = []
        for i in examples:
            if float(jsonInfo[i].get(maxAttr)) <= maxMargin:
                examples_son.append(i)
            else:
                examples_son2.append(i)
        if len(examples_son) == 0:
            nodeRelation[str(nodeId)]['isGood'] = getMostType(examples)
            return nownode
        elif len(examples_son2) == 0:
            nodeRelation[str(nodeId)]['isGood'] = getMostType(examples)
            return nownode
        else:
            nextAttrs = attributes.remove(maxAttr)
            nodeId = TreeGenerate(examples_son, nextAttrs, nownode, nodeId, "<=" + str(maxMargin))
            nodeId = TreeGenerate(examples_son2, nextAttrs, nownode, nodeId, ">" + str(maxMargin))

    return nodeId
    



TreeGenerate([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],keys[1:-1],0,0,"")
print(nodeRelation)



