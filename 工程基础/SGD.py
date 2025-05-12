# 导入必要的库
import torch  # PyTorch 是一个强大的科学计算库，用于张量操作和深度学习
import matplotlib.pyplot as plt  # Matplotlib 用于绘图，这里用来绘制散点图
from sklearn import linear_model  # scikit-learn 提供机器学习工具，这里使用线性回归
import torch.nn as nn  # PyTorch 的神经网络模块，用于定义模型
import torch.optim as optim  # PyTorch 的优化器模块，用于优化模型参数

# 设置随机种子，确保每次运行代码时生成的随机数相同（可重现性）
torch.manual_seed(1024)

# 生成数据
# torch.linspace(start, end, steps) 生成从 start 到 end 的均匀分布的 steps 个点
x = torch.linspace(100, 300, 200)  # 生成 200 个点，从 100 到 300
# 标准化 x：(x - 均值) / 标准差，使 x 的均值为 0，标准差为 1
# 标准化有助于数据规范化，消除量纲影响，便于模型训练
x = (x - torch.mean(x)) / torch.std(x)
# 生成随机噪声，服从标准正态分布（均值 0，标准差 1），形状与 x 相同
epsilon = torch.randn(x.shape)
# 生成因变量 y，y = 10*x + 5 + 噪声，模拟一个带噪声的线性关系
# 10 是斜率，5 是截距，epsilon 模拟现实中的随机误差
y = 10 * x + 5 + epsilon

# 绘制散点图
# plt.scatter(x, y) 将 x 和 y 绘制为散点图，展示数据分布
plt.scatter(x, y)
# plt.show() 显示图形窗口，运行后会看到散点图
plt.show()

# 使用 scikit-learn 进行线性回归
# 创建线性回归模型对象
m = linear_model.LinearRegression()
# 拟合模型：x 需要是二维数组 (样本数, 特征数)，所以用 x.view(-1, 1) 重塑
# y 是目标变量，是一维的
m.fit(x.view(-1, 1), y)
# 输出拟合结果：m.coef_ 是斜率，m.intercept_ 是截距
# 理论上斜率应接近 10，截距接近 5，但因噪声存在会有轻微偏差
print(m.coef_, m.intercept_)  # 输出示例：(array([9.904093]), 4.9480944)

### 使用 PyTorch 实现随机梯度下降 (SGD) 的线性回归
# 定义一个简单的线性回归模型，继承自 nn.Module
class Linear(nn.Module):
    def __init__(self):
        # 初始化模型
        super().__init__()  # 调用父类 nn.Module 的初始化方法
        # 定义模型参数 a（斜率）和 b（截距），初始化为 0
        # nn.Parameter 标记这些张量为可优化的参数
        self.a = nn.Parameter(torch.zeros(()))  # 标量张量，初始值 0
        self.b = nn.Parameter(torch.zeros(()))  # 标量张量，初始值 0

    def forward(self, x):
        # 定义前向传播：给定输入 x，计算输出 y = a*x + b
        return self.a * x + self.b

    def string(self):
        # 返回模型的字符串表示，显示当前参数 a 和 b 的值
        # .item() 将单值张量转换为 Python 标量
        return f'y = {self.a.item():.2f}*x + {self.b.item():.2f}'

# 创建模型实例
model = Linear()
# 测试模型：输入 x，计算初始输出（此时 a=0, b=0，所以输出全为 0）
model(x)

# 查看模型参数：返回 a 和 b 的列表
list(model.parameters())  # 输出：[Parameter(0.), Parameter(0.)]

# 设置超参数
learning_rate = 1  # 学习率，控制参数更新的步长
batch_size = 20  # 批量大小，每次训练使用 20 个数据点
# 创建模型实例
model = Linear()
# 创建优化器：AdamW 是一种改进的优化算法，优化模型参数
# lr 是学习率，model.parameters() 提供要优化的参数 (a 和 b)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# 训练循环：进行 20 次迭代
for t in range(20):
    # 计算当前批次的索引，循环使用数据
    # (t * batch_size) % len(x) 确保索引不会越界
    ix = (t * batch_size) % len(x)
    # 取出当前批次的数据：xx 是输入，yy 是目标值
    xx = x[ix:ix + batch_size]
    yy = y[ix:ix + batch_size]
    # 前向传播：用模型预测当前批次的输出
    yy_pred = model(xx)
    # 计算损失：均方误差 (MSE)，即 (预测值 - 真实值)^2 的平均值
    loss = (yy - yy_pred).pow(2).mean()
    # 清空之前的梯度，防止梯度累积
    optimizer.zero_grad()
    # 反向传播：计算损失对参数 (a, b) 的梯度
    loss.backward()
    # 手动更新参数：param -= learning_rate * param.grad
    # 这部分代码实际上被 optimizer.step() 替代，但保留了手动更新的逻辑
    with torch.no_grad():  # 禁用梯度计算，提高效率
        for param in model.parameters():
            param -= learning_rate * param.grad  # 更新参数
            param.grad = None  # 清空梯度
    # 使用优化器更新参数（与手动更新重复，这里仅需 optimizer.step() 即可）
    optimizer.step()
    # 打印当前模型的参数，表示当前拟合的直线
    print(model.string())