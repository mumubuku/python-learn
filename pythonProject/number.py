import tensorflow as tf
import matplotlib.pyplot as plt

# 加载MNIST数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train, x_test = x_train / 255.0, x_test / 255.0


# 选择要绘制的图像的索引
index = 0  # 例如，绘制第一个图像

# 绘制图像
plt.imshow(x_test[index], cmap='gray')  # 使用灰度色彩映射
plt.title(f'Label: {y_test[index]}')    # 可选，显示图像对应的标签
plt.show()



# 构建神经网络模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # 输入层：将图像展平成28*28=784个像素的向量
    tf.keras.layers.Dense(128, activation='relu'),  # 隐藏层1：128个神经元，使用ReLU激活函数
    tf.keras.layers.Dropout(0.2),  # Dropout层，用于防止过拟合
    tf.keras.layers.Dense(10, activation='softmax')  # 输出层：10个神经元，使用softmax激活函数，用于多类别分类
])

# 编译模型
model.compile(optimizer='adam',  # 选择优化器
              loss='sparse_categorical_crossentropy',  # 选择损失函数
              metrics=['accuracy'])  # 选择评估指标

# 训练模型并保存历史记录
history = model.fit(x_train, y_train, epochs=100)  # 训练10个epoch

# 评估模型
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# 绘制训练过程中的损失曲线
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
