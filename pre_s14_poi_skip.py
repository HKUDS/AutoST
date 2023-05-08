
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import pickle


def load_data(file):
    data_load_file = []
    file_1 = open(file, "rb")
    data_load_file = pickle.load(file_1)
    return data_load_file
poi_list = ['drinking_water', 'toilets', 'school', 'hospital', 'arts_centre', 'fire_station', 'police', 'bicycle_parking', 'fountain', 'ferry_terminal', 'bench', 'cinema', 'cafe', 'pub', 'waste_basket', 'parking_entrance', 'parking', 'fast_food', 'bank', 'restaurant', 'ice_cream', 'pharmacy', 'taxi', 'post_box', 'atm', 'nightclub', 'social_facility', 'bar', 'biergarten', 'clock', 'bicycle_rental', 'community_centre', 'watering_place', 'ranger_station', 'boat_rental', 'recycling', 'payment_terminal', 'bicycle_repair_station', 'place_of_worship', 'shelter', 'telephone', 'clinic', 'dentist', 'vending_machine', 'theatre', 'charging_station', 'public_bookcase', 'post_office', 'fuel', 'doctors']
poi_list_1 = ['drinking_water', 'toilets', 'school', 'hospital', 'arts_centre', 'fire_station', 'police', 'bicycle_parking', 'fountain', 'ferry_terminal', 'bench', 'cinema', 'cafe', 'pub', 'waste_basket', 'parking_entrance', 'parking', 'fast_food', 'bank', 'restaurant', 'ice_cream', 'pharmacy', 'taxi', 'post_box', 'atm', 'nightclub', 'social_facility', 'bar', 'biergarten', 'clock', 'bicycle_rental', 'community_centre', 'watering_place', 'ranger_station', 'boat_rental', 'recycling', 'payment_terminal', 'bicycle_repair_station', 'place_of_worship', 'shelter', 'telephone', 'clinic', 'dentist', 'vending_machine', 'theatre', 'charging_station', 'public_bookcase', 'post_office', 'fuel', 'doctors','drinking_water', 'toilets']
region_back = load_data("../data/region_back.pickle")
reg_poi = load_data("../data/reg_incld_poi_new.pickle")
# print(reg_poi)
# print(reg_poi)
poi_dict = {}
for idx, item in enumerate(poi_list):
    poi_dict[item] = idx
# print(poi_dict)
# println()



CONTEXT_SIZE = 2
EMBEDDING_DIM = 96  # 编码向量的维度

# test_sentence = """0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 0 1""".split()
test_sentence = poi_list_1
# print(test_sentence)
# preinln()
# 构建训练集数据 ([ 第一个单词, 第二个单词 ], 预测目标)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# trigrams = [([test_sentence[i]], test_sentence[i + 1])
#             for i in range(len(test_sentence) - 2)]
# print(trigrams)
# println()
# 构建测试集数据
vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
# print(vocab)
# print(word_to_ix)
# println()
# 定义模型
class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, EMBEDDING_DIM)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))  # 进行embedding
        # print(embeds.size())
        # pritjnln()
        out = F.relu(self.linear1(embeds))  # 经过第一个全连接层
        out = self.linear2(out)  # 经过第二个全连接层
        log_probs = F.log_softmax(out, dim=1)
        return log_probs,out

# # 进行训练
losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.0005)
emb_dict={}
for epoch in range(1500):
    total_loss = 0
    for context, target in trigrams:
        # 准备输入模型的数据
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        # print("context_idxs：",context_idxs)
        # print(context_idxs.size())
        # println()
        model.zero_grad()  # 清零梯度缓存

        # 进行训练得到预测结果
        log_probs,out = model(context_idxs)
        # print(out.size())
        # print("----:", out)
        # println()

        # 计算损失值
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # 反向传播更新梯度
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # 累计损失
        torch.save(model.state_dict(), './model_skip/model_poi.pt')
        torch.save(model, './model_skip/model_poi.pth')
        
        emb_dict[target] = out
    losses.append(total_loss)
print(losses)
# print(emb_dict)

poi_skip_vec = {}
for key,value in emb_dict.items():
    poi_skip_vec[poi_dict[key]] = torch.squeeze(value,0)
    # print("poi_dict[key]:",poi_dict[key])
    # print("size():", value.size())
# region_spatial = {}
# for key,value in reg_t_con.items():
#     # print(value)
#     region_spatial[key] = emb_dict[str(value)]
# print("---finish---:",len(region_spatial))
file=open(r"../data/poi_skip_vec.pickle","wb")
pickle.dump(poi_skip_vec,file) #storing_list
file.close()



