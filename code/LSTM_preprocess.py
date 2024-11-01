#导入程序所需要的包

import os #创建文件路径等操作使用
import subprocess #解析mid格式音乐文件时使用
import pickle
from pickle import dump, load #将提取好的变量存储时使用（因为解析非常耗时，节省时间）
import glob #按路径批量调用文件时使用
from music21 import converter,instrument,note,chord,stream#converter负责转换,乐器，音符，和弦类，解析时使用
import numpy as np

print('import OK!')

#定义批量解析所有乐曲使用的函数
def get_notes():
    """ 
    从musics目录中的所有MIDI文件里读取note，chord
    Note样例：B4，chord样例[C3,E4,G5],多个note的集合，统称“note”
    返回的为
    seqs:所有音乐序列组成的一个嵌套大列表(list)
    musicians:数据中所涉及的所有音乐家（不重复）
    namelist:按照音乐序列存储顺序与之对应的每一首乐曲的作曲家
    """
    seqs = []
    musicians = []
    namelist=[]
    #借助glob包获得某一路径下指定形式的所有文件名
    print(len(glob.glob("musics/*.mid")))
    i=1
    for file in glob.glob("musics/*.mid"):
        print(i)
        i += 1
        #提取出音乐家名(音乐家名包含在文件名中)
        name = file[7:-4].split(' (')
        #将新音乐家加入音乐家列表
        if name[0] not in musicians :
            musicians.append(name[0])  
        #初始化存放音符的序列
        notes = []
        #读取musics文件夹中所有的mid文件,file表示每一个文件
        #这里小部分文件可能解析会出错，我们使用python的try语句予以跳过
        try:
            stream=converter.parse(file)#midi文件的读取，解析，输出stream的流类型
        except:
            continue
        #获取所有的乐器部分
        parts=instrument.partitionByInstrument(stream)
        if parts:#如果有乐器部分，取第一个乐器部分
            try:
                notes_to_parse=parts.parts[1].recurse()#递归
            except:
                continue
        else:
            notes_to_parse=stream.flat.notes#纯音符组成
        for element in notes_to_parse:#notes本身不是字符串类型
            #这里考虑到后面的内存问题，对乐曲进行了分段处理
            if len(notes)<1000:
                #如果是note类型，取它的音高(pitch)
                if isinstance(element,note.Note):
                    #格式例如：E6
                    notes.append(str(element.pitch))
                elif isinstance(element,chord.Chord):
                    #转换后格式：45.21.78(midi_number)
                    notes.append('.'.join(str(n) for n in element.normalOrder))#用.来分隔，把n按整数排序
            else:
                seqs.append(notes)
                namelist.append(name[0])
                notes = []
        seqs.append(notes)
        namelist.append(name[0])
        
    return musicians,namelist,seqs#返回提取出来的notes列表


musicians,namelist,seqs = get_notes()
print('音乐家列表：')
print(musicians)
print('乐曲序列示例：')
print(seqs[0])
print('总乐曲个数')
print(len(seqs))


#我们在data文件夹下存储
# 如果 data 目录不存在，创建此目录
if not os.path.exists("data"):
    os.mkdir("data")
#将数据使用pickle写入文件
with open('data/seqs','wb') as filepath:#从路径中打开文件，写入
    pickle.dump(seqs,filepath)#把音符序列写入到文件中
with open('data/musicians','wb') as filepath:#从路径中打开文件，写入
    pickle.dump(musicians,filepath)#把音乐家列表写入到文件中
with open('data/namelist','wb') as filepath:#从路径中打开文件，写入
    pickle.dump(namelist,filepath)#把音乐家列表（重复计）写入到文件中

print('save OK!')    
    
#读入上面保存好的变量
musicians = load(open('data/musicians', 'rb'))
namelist = load(open('data/namelist', 'rb'))
seqs = load(open('data/seqs', 'rb'))
#再次展示音乐序列的第一个序列检查是否有问题
print(seqs[0])