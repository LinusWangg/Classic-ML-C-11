import csv
import json
import math

csv_file = open('2.csv', 'r')
keys = csv_file.readline().strip('\n').split(',')

jsonInfo = []
jsonInfoTest = []

i = 1

while True:
    values = csv_file.readline().strip('\n').split(',')
    if values == ['']:
        break
    dic_temp = dict(zip(keys,values)) 
    if i <= 10:
        jsonInfo.append(dic_temp)
    else:
        jsonInfoTest.append(dic_temp)
    i += 1

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

def calculateGini(examples):
    if len(examples) == 0:
        return 0
    typeTrue = 0
    typeFalse = 0
    res = 1
    for i in examples:
        if int(jsonInfo[i].get('isGood')) == 0:
            typeFalse += 1
        else:
            typeTrue += 1
    typeFalse = typeFalse / len(examples)
    typeTrue = typeTrue / len(examples)
    res -= typeFalse * typeFalse
    res -= typeTrue * typeTrue
    return res

def calculateGiniindex(examples, attrs):
    res = 0
    resMargin = 0
    if attrs != 'density' and attrs != 'sugar':
        for attr in elem_list[attrs]:
            examples_son = []
            for i in examples:
                if jsonInfo[i].get(attrs) == attr:
                    examples_son.append(i)
            res += (len(examples_son) / len(examples)) * calculateGini(examples_son)
        return [res,0.0]
    else:
        res = 2
        margins = []
        margins2 = []
        for x in elem_list[attrs]:
            margins.append(float(x))
        margins = sorted(margins)
        for i in range(len(margins) - 1):
            margins2.append((margins[i]+margins[i+1])/2)
        for x in margins2:
            res2 = 0
            examples_son = []
            examples_son2 = []
            for i in examples:
                if float(jsonInfo[i].get(attrs)) < x:
                    examples_son.append(i)
                elif float(jsonInfo[i].get(attrs)) > x:
                    examples_son2.append(i)
            res2 += (len(examples_son) / len(examples)) * calculateGini(examples_son)
            res2 += (len(examples_son2) / len(examples)) * calculateGini(examples_son2)
            if res2 < res:
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
    
    minGiniindex = 1
    minAttr = ""
    minMargin = 0.0
    for attr in attributes:
        Giniindex = calculateGiniindex(examples, attr)
        if minGiniindex > Giniindex[0]:
            minGiniindex = Giniindex[0]
            minAttr = attr
            minMargin = Giniindex[1]
    nodeRelation[str(nodeId)]['attr'] = minAttr
    if minAttr != 'density' and minAttr != 'sugar':
        for elem in elem_list[minAttr]:
            examples_son = []
            for i in examples:
                if jsonInfo[i].get(minAttr) == elem:
                    examples_son.append(i)
            if len(examples_son) == 0:
                nodeRelation[str(nodeId)]['isGood'] = getMostType(examples)
                return nownode
            else:
                nextAttrs = []
                for x in attributes:
                    if x == minAttr:
                        continue
                    nextAttrs.append(x)
                nodeId = TreeGenerate(examples_son, nextAttrs, nownode, nodeId, elem)
    else:
        examples_son = []
        examples_son2 = []
        for i in examples:
            if float(jsonInfo[i].get(minAttr)) <= minMargin:
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
            nextAttrs = attributes.remove(minAttr)
            nodeId = TreeGenerate(examples_son, nextAttrs, nownode, nodeId, "<=" + str(minMargin))
            nodeId = TreeGenerate(examples_son2, nextAttrs, nownode, nodeId, ">" + str(minMargin))

    return nodeId
    



TreeGenerate([0,1,2,3,4,5,6,7,8,9],keys[1:-1],0,0,"")
print(nodeRelation)



