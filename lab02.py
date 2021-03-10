import tensorflow as tf

x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]

W = tf.Variable(2.9)
b = tf.Variable(0.5)

# hypothesis = W * x + b
hypothesis = W * x_data + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

#tf.reduce_mean()
# v = [1, 2, 3, 4]
# tf.reduce_mean(v) #2.5

#Gradient descent

#learning_rate initialize
learning_rate = 0.01

for i in range(100):
    #gradient descent
    with tf.GradientTape() as tape: #with 구문안의 변수 정보를 tape에 기록
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
    W_grad, b_grad = tape.gradient(cost, [W, b]) #cost함수의 w, b 에 대한 미분값
    #A.assign_sub(B) : A -= B
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    if i % 10 == 0: #10의 배수마다 cost 계산결과 출력
        print(i, W.numpy(), b.numpy(), cost)
