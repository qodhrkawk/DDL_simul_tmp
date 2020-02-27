# from multiprocessing import Pool
import tensorflow as tf
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import arguments
from flmodel import Flmodel
from blockchain import Transaction, Blockchain
from node import Node
from malicious import Malicious
import csv

if __name__ == "__main__":
    rNum = 0

    mx_loss = []
    mx_acc = []
    mn_loss = []
    mn_acc = []
    ag_loss = []
    ag_acc = []
    round_label = []

    for i in range(50):
        round_label.append(i)

    def create_model():
        mnist_model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        mnist_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        return Flmodel(mnist_model)

    def my_policy_update_model_weights(self: Node, peer_weights: dict):
        # get reputation
        reputation = self.get_reputation()
        if len(reputation) == 0:
            raise ValueError
        if len(reputation) != len(peer_weights):
            raise ValueError

        ids = list(reputation.keys())

        # Setting new reputation

        testNode = self

        my_eval = self.evaluate_model()
        my_eff = my_eval[1] - my_eval[0]



        for node in nodes :
            if node.id == self.id :
                continue

            testNode.set_model_weights(node.get_model_weights())
            met = testNode.evaluate_model()
            if met[1]-met[0] > my_eff-my_eff*(1/(100-rNum)):
            #if met[1]-met[0] > my_eff-my_eff*0.1:
                reputation[node.id] = 1
            else:
                reputation[node.id] = 0

        self.set_reputation(reputation)



        total_reputation = sum(reputation.values()) + 1





        # original weights
        origin_weights = self.get_model_weights()


        # set zero-filled NN layers
        new_weights = list()
        for layer in peer_weights[ids[0]]:
            new_weights.append(np.zeros(layer.shape))



        # calculate new_weights
        # TODO: threshold
        # TODO: comparison

        for i, layer in enumerate(new_weights):
            for id in ids:
                layer += peer_weights[id][i] * \
                         reputation[id] / total_reputation
            # because all reputations are 1. for example.
            layer += origin_weights[i] / total_reputation

            # set new_weights
        self.set_model_weights(new_weights)


    def equally_fully_connected(my_id: str, ids: list):
        reputation = dict()
        for id in ids:
            if id == my_id:
                continue
            reputation[id] = 1.
        return reputation

    def my_policy_update_txs_weight(self: Blockchain, id: str):
        amount = self.get_transaction_by_id(id).weight
        predecessors_ids = self.get_all_predecessors_by_id(id)
        for p_id in predecessors_ids:
            p_tx = self.get_transaction_by_id(p_id)
            p_tx.weight += amount

    def avg_time(times):
        if len(times) == 0:
            return 0.0
        else:
            return sum(times) / len(times)

    """main"""

    # arguments
    args = arguments.parser()
    num_nodes = args.nodes
    num_round = args.round
    print("> Setting:", args)

    # load data
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0



    # split dataset
    # +1 for master testset
    my_x_train = np.array_split(x_train, (num_nodes + 1))
    my_y_train = np.array_split(y_train, (num_nodes + 1))
    my_x_test = np.array_split(x_test, (num_nodes + 1))
    my_y_test = np.array_split(y_test, (num_nodes + 1))




    master_testset_X = np.concatenate((my_x_train[-1], my_x_test[-1]))
    master_testset_Y = np.concatenate((my_y_train[-1], my_y_test[-1]))
    my_x_train = my_x_train[:-1]
    my_y_train = my_y_train[:-1]
    my_x_test = my_x_test[:-1]
    my_y_test = my_y_test[:-1]

    # set nodes
    ids = [str(i) for i in range(num_nodes)]
    nodes = list()
    for i, id in enumerate(ids):
        if i < (num_nodes-70):
            nodes.append(Node(
                id,
                create_model(),
                my_x_train[i], my_y_train[i],
                my_x_test[i], my_y_test[i],
                equally_fully_connected(id, ids),
                policy_update_model_weights_name="equal",
                policy_update_model_weights_func=my_policy_update_model_weights))

        else:

             my_new_y_train = np.zeros(len(my_y_train[i]))
             for j in range(len(my_y_train[i])):
                 if my_y_train[i][j] == 2:
                    my_new_y_train[j] = 4
                 else:
                    my_new_y_train[j] = my_y_train[i][j]

             nodes.append(Malicious(
                id,
                create_model(),
                my_x_train[i], my_new_y_train,
                my_x_test[i], my_y_test[i],
                equally_fully_connected(id, ids),
                policy_update_model_weights_name="equal",
                policy_update_model_weights_func=my_policy_update_model_weights))


    # set blockchain
    genesis_transaction = Transaction(
        nodes[0].id,
        int(time.time()),
        flmodel=nodes[0].get_model())  # node 0
    blockchain = Blockchain(
        genesis_transaction,
        policy_update_txs_weight_name="heaviest",
        policy_update_txs_weight_func=my_policy_update_txs_weight)
    # blockchain.print()

    # round
    # TODO: GPU
    # TODO: websocket
    elapsed_times = list()
    for r in range(num_round):
        start_time = time.time()

        print("Round", r)
        rNum = rNum+1
        # train
        for node in nodes:
            #if int(node.id) < 30:
            node.fit_model(epochs=1)

            # send transaction
            # for example: NO references
            tx = Transaction(
                node.id,
                int(time.time()),
                flmodel=node.get_model())
            blockchain.add_transaction(tx)

            print("train :\t node: %5s" % (node.id), end="\r")
        print(" " * 73, end="\r")

        # test
        losses = list()
        accuracies = list()
        for node in nodes:
            metrics = node.evaluate_model()
            # print("test  :\t", node.id, loss, metrics)
            losses.append(metrics[0])
            accuracies.append(metrics[1])
            print("own:\tnode: %5s\tloss: %7.4f\tacc : %7.4f," % (
                node.id, metrics[0], metrics[1]), end="\r")

        print(" " * 73, end="\r")
        print("own:\tmax_loss: %7.4f\tmin_loss: %7.4f\tavg_loss: %7.4f" % (
            max(losses), min(losses), sum(losses) / len(losses)))
        print("own:\tmax_acc : %7.4f\tmin_acc : %7.4f\tavg_acc : %7.4f" % (
            max(accuracies), min(accuracies), sum(accuracies) / len(accuracies)))

        # update weights
        for node in nodes:
            # get neighbors weights
            peer_weights = dict()
            for peer in nodes:
                if peer.id == node.id:
                    continue
                # peer_weights[peer.id] = peer.get_model_weights()
                peer_weights[peer.id] = blockchain.get_latest_transaction_by_owner(
                    peer.id).flmodel.get_weights()

            # TODO: duplicated weight update problem on the earliest nodes.
            node.update_model_weights(node, peer_weights)


            print("update:\t node: %5s" % (node.id), end="\r")
        print(" " * 73, end="\r")

        # test
        losses = list()
        accuracies = list()
        for node in nodes:
            metrics = node.evaluate_model()
            # print("update:\t", node.id, loss, metrics)
            losses.append(metrics[0])
            accuracies.append(metrics[1])
            print("mix:\tnode: %5s\tloss: %7.4f\tacc : %7.4f," % (
                node.id, metrics[0], metrics[1]), end="\r")

        print(" " * 73, end="\r")
        print("mix:\tmax_loss: %7.4f\tmin_loss: %7.4f\tavg_loss: %7.4f" % (
            max(losses), min(losses), sum(losses) / len(losses)))
        print("mix:\tmax_acc : %7.4f\tmin_acc : %7.4f\tavg_acc : %7.4f" % (
            max(accuracies), min(accuracies), sum(accuracies) / len(accuracies)))

        # eval. by master testset
        losses = list()
        accuracies = list()
        for node in nodes:
            metrics = node.get_model().evaluate(
                master_testset_X,
                master_testset_Y)
            # print("test  :\t", node.id, loss, metrics)
            losses.append(metrics[0])
            accuracies.append(metrics[1])
            print("mst:\tnode: %5s\tloss: %7.4f\tacc : %7.4f," % (
                node.id, metrics[0], metrics[1]), end="\r")

        print(" " * 73, end="\r")
        print("mst:\tmax_loss: %7.4f\tmin_loss: %7.4f\tavg_loss: %7.4f" % (
            max(losses), min(losses), sum(losses) / len(losses)))
        mx_loss.append(max(losses))
        mn_loss.append(min(losses))
        ag_loss.append(sum(losses) / len(losses))
        print("mst:\tmax_acc : %7.4f\tmin_acc : %7.4f\tavg_acc : %7.4f" % (
            max(accuracies), min(accuracies), sum(accuracies) / len(accuracies)))
        mx_acc.append(max(accuracies))
        mn_acc.append(min(accuracies))
        ag_acc.append(sum(accuracies) / len(accuracies))


        # time
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_times.append(elapsed_time)
        print("elapsed time: %f\tETA: %f" %
              (elapsed_time, avg_time(elapsed_times)))




    mx_data = []
    mn_data = []
    ag_data = []
    for i in range(50):
        mx_data.append([mx_acc[i],mx_loss[i]])
        mn_data.append([mn_acc[i],mn_loss[i]])
        ag_data.append([ag_acc[i],ag_loss[i]])


    max_csvfile = open("max.csv", "w", newline="")
    max_csvwriter = csv.writer(max_csvfile)
    for row in mx_data:
        max_csvwriter.writerow(row)
    max_csvfile.close()

    min_csvfile = open("min.csv", "w", newline="")
    min_csvwriter = csv.writer(min_csvfile)


    for row in mn_data :
        min_csvwriter.writerow(row)

    min_csvfile.close()

    avg_csvfile = open("avg.csv", "w", newline="")
    avg_csvwriter = csv.writer(avg_csvfile)
    for row in ag_data:
        avg_csvwriter.writerow(row)
    avg_csvfile.close()


    print(mx_data)

    with open("max.csv", "r", newline="") as max_csvfile:
        max_data_reader = csv.reader(max_csvfile)
        max_csv_data = [row for row in max_data_reader]
    max_accs = [float(elem[0]) for elem in max_csv_data]
    max_losses = [float(elem[1]) for elem in max_csv_data]

    max_accs = np.array(max_accs)
    max_losses = np.array(max_losses)
    fig, ax = plt.subplots()

    print(max_accs)
    print(max_losses)

    ax.plot(round_label, max_accs)
    ax.plot(round_label, max_losses)
    plt.show()

    with open("min.csv", "r", newline="") as min_csvfile:
        min_data_reader = csv.reader(min_csvfile)
        min_csv_data = [row for row in min_data_reader]
    min_accs = [float(elem[0]) for elem in min_csv_data]
    min_losses = [float(elem[1]) for elem in min_csv_data]

    min_accs = np.array(min_accs)
    min_losses = np.array(min_losses)
    fig, ax = plt.subplots()
    ax.plot(round_label, min_accs)
    ax.plot(round_label, min_losses)
    plt.show()

    with open("avg.csv", "r", newline="") as avg_csvfile:
        avg_data_reader = csv.reader(avg_csvfile)
        avg_csv_data = [row for row in avg_data_reader]
    avg_accs = [float(elem[0]) for elem in avg_csv_data]
    avg_losses = [float(elem[1]) for elem in avg_csv_data]

    avg_accs = np.array(avg_accs)
    avg_losses = np.array(avg_losses)
    fig, ax = plt.subplots()
    ax.plot(round_label, avg_accs)
    ax.plot(round_label, avg_losses)
    plt.show()

