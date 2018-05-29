'''This .py file builds the training loop using the training parameters'''

# Importing the necessary packages
from imports import *
from layers_info import *


# Placeholder to store the actual probability of each class for each input image
y_true=tf.placeholder(tf.float32,[None,num_classes],name = 'y_true')

y_true_cls = tf.argmax(y_true, dimension=1)

# Defining cost that will be minimized to reach the optimum value of weights
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = y_pred, labels = y_true, name = 'pred')
total_loss = tf.reduce_mean(cross_entropy)

# Define the training operation
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

# Operation comparing prediction with true label
correct_prediction=tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

# Operation calculating the accuracy of our predictions
accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Tensorboard
tf.summary.scalar("loss", total_loss)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)
# Create a summary to monitor weights and biases present in the network
tf.summary.histogram("W_conv1",W_conv1)
tf.summary.histogram("b_conv1",b_conv1)
tf.summary.histogram("W_conv3",W_conv3)
tf.summary.histogram("b_conv3",b_conv3)
tf.summary.histogram("W_conv5",W_conv5)
tf.summary.histogram("b_conv5",b_conv5)
tf.summary.histogram("W_fc_7",W_fc_7)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()
logs_path = '/tmp/tensorflow_logs/example/'
#  Training loop
saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

# Initializing the lists required to store the loss and accuracy
train_acc_list = list()
val_acc_list = list()
train_loss_list = list()
val_loss_list = list()
epochs_list = list()

with tf.Session() as sess:
    # Initialize variables
    sess.run(init_op)
    num_iterations = num_epochs * int(data.train.num_examples/training_batch_size)
    # Tensorboard summary function
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for i in range(num_iterations):
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(training_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(training_batch_size)
        feed_dict_tr = {images_placeholder: x_batch,y_true: y_true_batch}
        feed_dict_val = {images_placeholder: x_valid_batch, y_true: y_valid_batch}
        sess.run(optimizer, feed_dict = feed_dict_tr)
        # Tensorboard
        _, c, summary = sess.run([optimizer,total_loss,merged_summary_op],feed_dict=feed_dict_tr)
        summary_writer.add_summary(summary, i)
        if i % int(data.train.num_examples/training_batch_size) == 0:
            epoch = int(i / int(data.train.num_examples/training_batch_size))

            #Calculating training and validation accuracy
            acc = sess.run(accuracy, feed_dict = feed_dict_tr)
            val_acc = sess.run(accuracy, feed_dict = feed_dict_val)
            val_loss = sess.run(total_loss, feed_dict = feed_dict_val)
            train_loss = sess.run(total_loss, feed_dict = feed_dict_tr)
            msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%},Validation Accuracy: {2:>6.1%},Validation Loss: {3:.3f},Training Loss: {4:.3f}"
            print(msg.format(epoch + 1, acc, val_acc, val_loss, train_loss))

            epochs_list.append(epoch + 1)
            train_acc_list.append(acc*100)
            val_acc_list.append(val_acc*100)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            if len(epochs_list) >= 2:
                # Plotting the training accuracy inference
                plt.subplot(2,2,1)
                plt.plot(epochs_list,train_acc_list, label = "epoch" + str(epoch + 1))
                plt.xlabel('Epoch')
                plt.xlim((1, num_epochs + 1))
                plt.ylabel('Training accuracy(%)')
                plt.ylim((0, 101))
                plt.title('Training accuracy inference')
                plt.legend()
                plt.subplot(2,2,2)
                # Plotting the Validation accuracy inference
                plt.plot(epochs_list,val_acc_list, label = "epoch" + str(epoch + 1))
                plt.xlabel('Epoch')
                plt.xlim((1, num_epochs + 1))
                plt.ylabel('Validation accuracy(%)')
                plt.ylim((0, 101))
                plt.title('Validation accuracy inference')
                plt.legend()
                # Plotting the training loss inference
                plt.subplot(2,2,3)
                plt.plot(epochs_list,train_loss_list, label = "epoch" + str(epoch + 1))
                plt.xlabel('Epoch')
                plt.xlim((1, num_epochs + 1))
                plt.ylabel('Training loss')
                plt.title('Training loss inference')
                plt.legend()
                # Plotting the validation loss inference
                plt.subplot(2,2,4)
                plt.plot(epochs_list,val_loss_list, label = "epoch" + str(epoch + 1))
                plt.xlabel('Epoch')
                plt.xlim((1, num_epochs + 1))
                plt.ylabel('Validation loss')
                plt.title('Validation loss inference')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(target_path, "epoch.png"))
                #plt.show(block = False)
                time.sleep(3)
                plt.close()
            # Saving the model after training
            save_path = saver.save(sess, r"/home/saikantareddy/Documents/eAI_Generator/eAI_generator_integrated_code/model/generator")

