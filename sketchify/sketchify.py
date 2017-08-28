# Main sketchify tool.
# Base code from: https://github.com/kvfrans/deepcolor/blob/master/main.py

import math
import os
import sys
import random
from glob import glob

try:
    from sketchify.sketchify_utils import *
except ImportError:
    from sketchify_utils import *
# from collect_anime_pics import resize_dim

random.seed(42)

NUM_ITER = 216 # Was 20k before
SAVE_INTERVALS = 5


class Sketchify():
    def __init__(self, imgsize=256, batchsize=16):
        self.batch_size = batchsize
        self.batch_size_sqrt = int(math.sqrt(self.batch_size))
        self.image_size = imgsize
        self.output_size = imgsize

        self.gf_dim = 64
        self.df_dim = 64

        self.input_colors = 1
        self.input_colors2 = 1
        self.output_colors = 1

        self.l1_scaling = 100

        # with tf.variable_scope(tf.get_variable_scope()) as scope:

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.binary_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_colors])
        # self.color_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_colors2])
        self.sketch_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.output_colors])

        # combined_preimage = tf.concat([self.line_images, self.color_images], 3)
        # combined_preimage = self.line_images

        self.generated_images = self.generator(self.binary_images)

        self.real_AB = tf.concat([self.binary_images, self.sketch_images], 3)
        self.fake_AB = tf.concat([self.binary_images, self.generated_images], 3)

        self.disc_true, disc_true_logits = self.discriminator(self.real_AB, reuse=False)
        self.disc_fake, disc_fake_logits = self.discriminator(self.fake_AB, reuse=True)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_true_logits, labels=tf.ones_like(disc_true_logits)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=tf.zeros_like(disc_fake_logits)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=tf.ones_like(disc_fake_logits))) \
                        + self.l1_scaling * tf.reduce_mean(tf.abs(self.sketch_images - self.generated_images))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)

    def discriminator(self, image, y=None, reuse=False):
        # Error: no variable d_h0_conv/w/Adam
        # https://github.com/carpedm20/DCGAN-tensorflow/commit/6c2a0ca5241eed7c83b7c38c0e46450b9a77fc3d
        with tf.variable_scope("discriminator") as scope:
            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv')) # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv'))) # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv'))) # h2 is (32 x 32 x self.df_dim*4)
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv'))) # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
            return tf.nn.sigmoid(h4), h4

    def generator(self, img_in):
        s = self.output_size
        s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(img_in, self.gf_dim, name='g_e1_conv') # e1 is (128 x 128 x self.gf_dim)
        e2 = bn(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv')) # e2 is (64 x 64 x self.gf_dim*2)
        e3 = bn(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv')) # e3 is (32 x 32 x self.gf_dim*4)
        e4 = bn(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv')) # e4 is (16 x 16 x self.gf_dim*8)
        e5 = bn(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv')) # e5 is (8 x 8 x self.gf_dim*8)


        self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(e5), [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
        d4 = bn(self.d4)
        d4 = tf.concat([d4, e4], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4), [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
        d5 = bn(self.d5)
        d5 = tf.concat([d5, e3], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5), [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
        d6 = bn(self.d6)
        d6 = tf.concat([d6, e2], 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6), [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
        d7 = bn(self.d7)
        d7 = tf.concat([d7, e1], 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7), [self.batch_size, s, s, self.output_colors], name='g_d8', with_w=True)
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(self.d8)

    def imageblur(self, cimg, sampling=False):
        if sampling:
            cimg = cimg * 0.3 + np.ones_like(cimg) * 0.7 * 255
        else:
            for i in range(30):
                randx = random.randint(0, 205)
                randy = random.randint(0, 205)
                cimg[randx:randx+50, randy:randy+50] = 255
        return cv2.blur(cimg, (100, 100))

    def train(self):
        self.loadmodel()

        binary_file_training = glob(os.path.join("../mangify_256_256", "*.jpg"))
        sketch_file_training = glob(os.path.join('../sketchify_256_256', '*.jpg'))
        assert (len(binary_file_training) == len(sketch_file_training))

        binary_file_validation = glob(os.path.join("validation/mangify", "*.jpg"))
        sketch_file_validation = glob(os.path.join("validation/sketchify", "*.jpg"))
        assert (len(binary_file_validation) == len(sketch_file_validation))


        binary_file_training.sort()
        sketch_file_training.sort()
        print('Binary:', binary_file_training[0])
        print('Sketch:', sketch_file_training[0])
        assert (all([binary_img.split('\\')[1] == sketch_img.split('\\')[1]
                    for binary_img, sketch_img in zip(binary_file_training, sketch_file_training)])
                and len(sketch_file_training) >= 100)

        binary_file_validation.sort()
        sketch_file_validation.sort()
        assert (all([binary_img.split('\\')[1] == sketch_img.split('\\')[1]
                     for binary_img, sketch_img in zip(binary_file_validation, sketch_file_validation)])
                and len(sketch_file_validation) >= 4)

        # print(binary_file_training[0])
        binary_imgs_training = np.array([get_image(sample_file, grayscale=True)
                                for sample_file in binary_file_training])
        binary_imgs_training = np.expand_dims(binary_imgs_training, axis=3) / 255.0 # 4 x 256 x 256 x 1, with grayscale values in [0,1]
        sketch_imgs_training = np.array([get_image(sample_file, grayscale=True)
                                for sample_file in sketch_file_training])
        sketch_imgs_training = np.expand_dims(sketch_imgs_training, axis=3) / 255.0

        print('Sketch image:', sketch_imgs_training.shape)
        # print('Sketch image:', np.expand_dims(sketch_imgs_training, axis=3).shape)

        # base_edge = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        #                                             cv2.THRESH_BINARY, blockSize=9, C=2) for ba in base]) / 255.0
        # base_edge = np.expand_dims(base_edge, 3)

        # base_colors = np.array([self.imageblur(ba) for ba in base]) / 255.0
        # binary_imgs_validation, sketch_imgs_validaion = \
        #     binary_imgs_training[0:self.batch_size], sketch_imgs_training[0:self.batch_size]

        binary_imgs_validation = np.array([get_image(sample_file, grayscale=True)
                                              for sample_file in binary_file_validation])
        binary_imgs_validation = np.expand_dims(binary_imgs_validation, axis=3) / 255.0
        sketch_imgs_validaion = np.array([get_image(sample_file, grayscale=True)
                                              for sample_file in sketch_file_validation])
        sketch_imgs_validaion = np.expand_dims(sketch_imgs_validaion, axis=3) / 255.0


        ims("results/base_binary.jpg", merge_grayscale(binary_imgs_validation, [self.batch_size_sqrt, self.batch_size_sqrt]))
        ims("results/base_sketch.jpg", merge_grayscale(sketch_imgs_validaion, [self.batch_size_sqrt, self.batch_size_sqrt]))
        # ims("results/base_colors.jpg", merge_color(base_colors, [self.batch_size_sqrt, self.batch_size_sqrt]))

        datalen = len(binary_imgs_training)
        print('----------------------------------------------------')
        print('Number of training images, batches:', datalen, datalen // self.batch_size)

        start_time = time.time()
        print('Starting time for %d iters:' % (NUM_ITER,))
        print(convert_to_datetime(start_time))
        for e in range(1, NUM_ITER + 1):
            recorded_recreation = False
            recorded_checkpoint = False
            k = datalen // self.batch_size
            for i in range(1, k + 1):
                train_binary_batch = binary_imgs_training[(i-1)*self.batch_size:(i)*self.batch_size]
                truth_sketch_batch = sketch_imgs_training[(i-1)*self.batch_size:(i)*self.batch_size]
                # batch = np.array([get_image(batch_file) for batch_file in batch_files])
                # batch_normalized = batch / 255.0
                #
                # batch_edge = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                #                                              cv2.THRESH_BINARY, blockSize=9, C=2) for ba in batch]) / 255.0
                # batch_edge = np.expand_dims(batch_edge, 3)



                # batch_colors = np.array([self.imageblur(ba) for ba in batch]) / 255.0

                d_loss, _ = self.sess.run([self.d_loss, self.d_optim],
                                          feed_dict={self.sketch_images: truth_sketch_batch,
                                                     self.binary_images: train_binary_batch})
                g_loss, _ = self.sess.run([self.g_loss, self.g_optim],
                                          feed_dict={self.sketch_images: truth_sketch_batch,
                                                     self.binary_images: train_binary_batch})

                print("%d: [%d / %d] d_loss %f, g_loss %f" % (e, i, (k), d_loss, g_loss))

                if (e % 5 == 0 or e == NUM_ITER) and (i == k) and not recorded_recreation:
                    recorded_recreation = True
                    recreation, d_loss_v = self.sess.run([self.generated_images, self.d_loss],
                                               feed_dict={self.sketch_images: binary_imgs_validation,
                                                          self.binary_images: sketch_imgs_validaion})

                    ims("results/"+str(e*100000 + i)+".jpg",merge_grayscale_2x2(recreation, [self.batch_size_sqrt, self.batch_size_sqrt]))
                    print('Cross-validation loss:', d_loss_v)

                if i == k and (e % (NUM_ITER // 27) == 0 or e == NUM_ITER) and not recorded_checkpoint:
                    recorded_checkpoint = True
                    self.save("./checkpoint", e*100000 + i)
        end_time = time.time()
        print('Start time:', convert_to_datetime(start_time))
        print('End time:', convert_to_datetime(end_time))

    def loadmodel(self, load_discrim=True):
        # Error: Could not create cuDNN handle
        # https://github.com/tensorflow/tensorflow/issues/6698
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.sess.run(tf.initialize_all_variables())

        if load_discrim:
            self.saver = tf.train.Saver()
        else:
            self.saver = tf.train.Saver(self.g_vars)

        if self.load("./checkpoint"):
            print("Loaded")
        else:
            print("Load failed")

    def sample(self, use='validation'):
        # if use != 'validation':
        #     use = 'testing'
        saveuse = use
        self.loadmodel(False)

        data = glob(os.path.join("%s/mangify" % use, "*.jpg"))[:min(self.batch_size, 16)]

        datalen = len(data)
        print(data)

        # for i in range(min(100, datalen // self.batch_size)):
        binary_image_batch = data
        batch = np.array([cv2.resize(imread(batch_file, grayscale=True), (256,256))
                          for batch_file in data])
        batch_normalized = np.expand_dims(batch, axis=3) / 255.0

        # batch_edge = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
        #                        for ba in batch]) / 255.0
        # batch_edge = np.expand_dims(batch_edge, 3)
        #
        # batch_colors = np.array([self.imageblur(ba,True) for ba in batch]) / 255.0

        recreation = self.sess.run(self.generated_images, feed_dict={self.binary_images: batch_normalized})
        ims(("%s/sample_recreaction" % saveuse) +str(0)+".png",merge_grayscale(recreation, [self.batch_size_sqrt, self.batch_size_sqrt]))
        # ims("validation/sample_original" + str(0) + ".jpg",
        #     merge_grayscale(batch, [self.batch_size_sqrt, self.batch_size_sqrt]))

    def save(self, checkpoint_dir, step):
        model_name = "model"
        model_dir = "tr"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "tr"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main.py [train, sample]")
    else:
        cmd = sys.argv[1]
        if cmd == "train":
            c = Sketchify()
            c.train()
        elif cmd == "sample":
            c = Sketchify(256, 4)
            c.sample(use='testing')
        else:
            print("Usage: python main.py [train, sample]")