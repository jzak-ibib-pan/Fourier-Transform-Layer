from tensorflow.keras import backend as K


def session_reset():
    K.clear_session()


if __name__ == '__main__':
    print(0)