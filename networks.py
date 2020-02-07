# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import gen_linalg_ops

# NOTE: Do not import any application-specific modules here!

#----------------------------------------------------------------------------

def lerp(a, b, t): return a + (b - a) * t
def lerp_clip(a, b, t): return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
def cset(cur_lambda, new_cond, new_lambda): return lambda: tf.cond(new_cond, new_lambda, cur_lambda)

#----------------------------------------------------------------------------
def batch_sqrtm(A, numIters = 10, reg = 2.0, caps_type=100,dim=4):
    """
    Batch matrix root via Newton-Schulz iterations
    """
    #batchSize = A.shape[0]
    #dim = A.shape[1]
    #Renormalize so that the each matrix has a norm lesser than 1/reg, but only normalize when necessary

    normA=reg* tf.sqrt(1e-6 + tf.reduce_sum(tf.square(A), axis=[1,2], keepdims=True))
    #print('I am here-1',normA.get_shape())
    ones=tf.ones_like(normA)
    renorm_factor=tf.where(normA>ones,normA,ones)
    #print('I am here0',renorm_factor.get_shape())
    Y=A/renorm_factor
    #print('I am here1',Y.get_shape())
    I=tf.eye(dim,dim,batch_shape=[caps_type])
    
    
    #I=I.reshape([1,dim,dim])
    #I= T.repeat(I,bs,axis=0)
    Z=tf.eye(dim,dim,batch_shape=[caps_type])#I.copy()
    for i in range(numIters):
        t=0.5*(3.0*I-tf.matmul(Z,Y))
        #t=0.5*(3.0*I-th.tensor.batched_dot(Z,Y))
        #Y=th.tensor.batched_dot(Y,t)        
        Y=tf.matmul(Y,t)
        Z=tf.matmul(t,Z)
        #Z=th.tensor.batched_dot(t,Z)
    sA=Y* tf.sqrt(renorm_factor)
    sAinv=Z/tf.sqrt(renorm_factor)
    return sA,sAinv


#----------------------------------------------------------------------------

# Get/create weight tensor for a convolutional or fully-connected layer.

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    #print('I am here in get_weight')
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

def get_weight_caps(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None,caps_dim=4,isconv=True,scale=True,iter=False):
    eps=1e-6
    #print('I am here in get_weight')
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init
    if isconv:
        caps_num=shape[3]//caps_dim
    else:
        caps_num=shape[2]//caps_dim
        shape=[shape[0],shape[1],shape[3],shape[2]]
    assert caps_dim*caps_num==shape[3]

    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        w= tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        w= tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))
    if scale:
        s=tf.get_variable('S',[caps_num,1,1],initializer=tf.constant_initializer(1.0))
    w=tf.reshape(w,[shape[0],shape[1],shape[2],caps_num,caps_dim]) # size = [[kernel_size,kernel_size,filters_in,caps_num,caps_dim]]
    w=tf.transpose(w,[3,2,0,1,4])  #size= [caps_num,filter_in, kernel_size_h,kernel_size_w,caps_dim]
    w=tf.reshape(w,[caps_num,shape[2]*shape[0]*shape[1],caps_dim]) # size= [caps_num,filters_in*kernel_size*kernel_size,caps_dim]
    sigma=tf.matmul(w,w,True,False)  # sigma size= [caps_num, caps_dim , caps_dim]
    if iter:
        sigma,sigma_sqrt=batch_sqrtm(sigma, numIters = 10, reg = 2.0, caps_type=caps_num,dim=caps_dim)
    else:
        sigma=tf.matrix_inverse(sigma+eps*tf.eye(caps_dim)) # sigma^-1 = [caps_num, caps_dim , caps_dim]
        sigma_sqrt=linalg_ops.matrix_square_root(sigma) # sigma^-1/2= [ caps_num, caps_dim, caps_dim ]
    #sigma_sqrt=gen_linalg_ops.matrix_square_root(sigma)
    w=tf.matmul(w,sigma_sqrt)  # size [caps_num,filters_in*kernel_size*kernel_size,caps_dim]
    if scale:
        w=w*s
    w=tf.reshape(w,[caps_num,shape[2],shape[0],shape[1],caps_dim])
    w=tf.transpose(w,[2,3,1,0,4])
    w=tf.reshape(w,[shape[0],shape[1],shape[2],caps_num*caps_dim])

    if not isconv:
        w=tf.transpose(w,[0,1,3,2])
    return w
def get_weight_caps_dense(shape,gain=np.sqrt(2),use_wscale=False,fan_in=None,scale=True):
    eps=1e-6
    #print('I am here in get_weight')
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        w= tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        w= tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

    if scale:
        s=tf.get_variable('S',[shape[0],1,1],initializer=tf.constant_initializer(1.0))
        return w,s
    return w,_
    #w= tf.get_variable('W_'+name, [lbl_num,input_dim,dim_c ], initializer=initializer)#tf.random_normal_initializer(stddev=stddev)) # starting from scratch

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense(x, fmaps, gain=np.sqrt(2), use_wscale=False):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    #print('I am here in dense')
    return tf.matmul(x, w)


def dense_caps(x,fmaps,gain=np.sqrt(2), use_wscale=False,caps_dim=4):
    eps=1e-8
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
        w,s = get_weight_caps_dense([fmaps,x.shape[1].value,caps_dim], gain=gain, use_wscale=use_wscale,fan_in=None,scale=False)
        sigma=tf.matmul(a=w,b=w,transpose_a=True,transpose_b=False) #size=[fmaps,dim_c,dim_c]
        sigma=tf.matrix_inverse(sigma+eps*tf.eye(caps_dim))
        x_tile=tf.einsum('bi,aij->baj',x,w) # size=[bs,fmaps,dim_c] # L to R
        x=tf.einsum('bai,aij->baj',x_tile,sigma) # size=[bs,fmaps,dim_c]
        x=tf.einsum('bai,bai->ba',x,x_tile) #size=[bs,fmaps]
        x=tf.sqrt(x+eps)
        return x
#----------------------------------------------------------------------------
# Convolutional layer.

def conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    #print('I am here in conv2d')
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')

def conv2d_caps(x, fmaps, kernel,caps_dim, gain=np.sqrt(2), use_wscale=False,scale=True,iter=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight_caps([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale,caps_dim=caps_dim,isconv=True,scale=scale,iter=iter)
    w = tf.cast(w, x.dtype)
    #print('I am here in conv2d_caps')
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')

def conv_dense_caps(x,fmaps,gain=np.sqrt(2), use_wscale=False,caps_dim=4):
    shape=x.get_shape()
    w=get_weight_caps([x.shape[2].value,x.shape[3].value,x.shape[1].value,fmaps],gain=gain,use_wscale=use_wscale,caps_dim=caps_dim)
    x=tf.nn.conv2d(x,w,[1,1,shape[2],shape[3]],padding='VALID',data_format='NCHW')
    return x




#----------------------------------------------------------------------------
# Apply bias to the given activation tensor.

def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1])

def apply_bias_caps(x,caps_dim):
    caps_num=x.shape[1]//caps_dim
    #print('888888888888888888',x.get_shape())
    x=tf.reshape(x,[-1,caps_num,caps_dim,x.shape[2],x.shape[3]])
    #print('888888888888888888',x.get_shape())
    b=tf.get_variable('bias', shape=[caps_num], initializer=tf.initializers.zeros())
    #print('bb888888888888888888',b.get_shape())
    x=x + tf.reshape(b, [1, -1,1, 1, 1])
    return tf.reshape(x,[-1,caps_num*caps_dim,x.shape[3],x.shape[4]])


#----------------------------------------------------------------------------
# Leaky ReLU activation. Same as tf.nn.leaky_relu, but supports FP16.

def leaky_relu(x, alpha=0.2):
    with tf.name_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.maximum(x * alpha, x)

#----------------------------------------------------------------------------
# Nearest-neighbor upscaling layer.

def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        #print('I am here in upscale2d')
        return x
#---------------------------------------------------------------------------
# bilinear upsampling layer
def bi_upscale2d(x):#, factor=2):
    #assert isinstance(factor, int) and factor >= 1
    #if factor == 1: return x
    with tf.variable_scope('Bi_Upscale2D'):
        s = x.shape
        x=tf.transpose(x,[0,2,3,1])#[-1,s[2],s[3],s[1]])
        x=tf.image.resize_bilinear(x,[2*s[2],2*s[3]])
        x=tf.transpose(x,[0,3,1,2])#s[1],2*s[2],2*s[3]])
        #x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        #x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        #x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        #print('I am here in upscale2d')
        return x



#----------------------------------------------------------------------------
# Fused upscale2d + conv2d.
# Faster and uses less memory than performing the operations separately.

def upscale2d_conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, fmaps, x.shape[1].value], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*x.shape[1].value)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
    #print('I am here in upscale2d_conv2d')
    return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')


def upscale2d_conv2d_caps(x, fmaps, kernel,caps_dim, gain=np.sqrt(2), use_wscale=False,scale=True,iter=False):#(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w =get_weight_caps([kernel, kernel, fmaps, x.shape[1].value], gain=gain, use_wscale=use_wscale,caps_dim=caps_dim,isconv=False,scale=scale,iter=iter)
    #get_weight([kernel, kernel, fmaps, x.shape[1].value], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*x.shape[1].value)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
    #print('I am here in upscale2d_conv2d_caps')
    return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Box filter downscaling layer.

def downscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Downscale2D'):
        ksize = [1, 1, factor, factor]
        #print('I am here in downsclae2d')
        return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW') # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True

#----------------------------------------------------------------------------
# Fused conv2d + downscale2d.
# Faster and uses less memory than performing the operations separately.

def conv2d_downscale2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    w = tf.cast(w, x.dtype)
    #print('I am here in conv2d_downscaled2d')
    return tf.nn.conv2d(x, w, strides=[1,1,2,2], padding='SAME', data_format='NCHW')


def conv2d_downscale2d_caps (x, fmaps, kernel,caps_dim, gain=np.sqrt(2), use_wscale=False,scale=True,iter=False):#(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight_caps([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale,caps_dim=caps_dim,isconv=True,scale=scale,iter=iter)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    w = tf.cast(w, x.dtype)
    #print('I am here in conv2d_downscaled2d_caps')
    return tf.nn.conv2d(x, w, strides=[1,1,2,2], padding='SAME', data_format='NCHW')



#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

def pixel_norm_caps(x, epsilon=1e-8, caps_dim=4):
    with tf.variable_scope('PixelNorm'):
        caps_num=x.shape[1].value//caps_dim
        x=tf.reshape(x,[-1,caps_num,caps_dim,x.shape[2].value,x.shape[3].value])
        scale=tf.reduce_mean(tf.square(x), axis=2, keepdims=True)
        #x= x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=2, keepdims=True) + epsilon)
        x=x*(tf.sqrt(scale)/(1+scale))
        x=tf.reshape(x,[-1,caps_num*caps_dim,x.shape[3].value,x.shape[4].value])
        return x

def score_act(x,epsilon=1e-8,caps_dim=4,th=0.3):
    with tf.variable_scope('Scoring'):
        caps_num=x.shape[1].value//caps_dim
        b = tf.get_variable('bias', shape=[caps_num], initializer=tf.constant_initializer(th))
        b = tf.cast(b, x.dtype)
        
        x=tf.reshape(x,[-1,caps_num,caps_dim,x.shape[2].value,x.shape[3].value])
        Norm=tf.sqrt( tf.reduce_sum(tf.square(x), axis=2, keepdims=True))
        
        scale= tf.nn.relu(Norm-tf.reshape(tf.square(b),[1,-1,1,1,1]))
        x=(scale/(Norm+epsilon))*x
        
        x=tf.reshape(x,[-1,caps_num*caps_dim,x.shape[3].value,x.shape[4].value])
        
        return x
        #if len(x.shape) == 2:
        #    return x + b
        #else:
        #    return x + tf.reshape(b, [1, -1, 1, 1])



#----------------------------------------------------------------------------
# Minibatch standard deviation.

def minibatch_stddev_layer(x, group_size=4):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [N1HW]  Replicate over group and pixels.
        return tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.

#----------------------------------------------------------------------------
# Batch normalization.
def bn(x,training=True):
    with tf.variable_scope('BatchNorm'):
        x=tf.layers.batch_normalization(x,axis=1,momentum=0.9,training=training)
        return x

def ln(x,training=True):
    with tf.variable_scope('LayerNorm'):
        x=tf.contrib.layers.layer_norm(x, begin_norm_axis=1,
    begin_params_axis=1)
        return x
#----------------------------------------------------------------------------
# Generator network used in the paper.

def G_paper(
    latents_in,                         # First input: Latent vectors [minibatch, latent_size].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    num_channels        = 1,            # Number of output color channels. Overridden based on dataset.
    resolution          = 32,           # Output resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    latent_size         = None,         # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
    normalize_latents   = True,         # Normalize latent vectors before feeding them to the network?
    use_wscale          = True,         # Enable equalized learning rate?
    use_pixelnorm       = True,         # Enable pixelwise feature vector normalization?
    pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
    use_leakyrelu       = True,         # True = leaky ReLU, False = ReLU.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = True,         # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    **kwargs):                          # Ignore unrecognized keyword args.
    
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def PN(x): return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x

    if latent_size is None: latent_size = nf(0)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu if use_leakyrelu else tf.nn.relu
    
    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    combo_in = tf.cast(tf.concat([latents_in, labels_in], axis=1), dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    # Building blocks.
    def block(x, res): # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            #print('**********************','%dx%d' % (2**res, 2**res))
            if res == 2: # 4x4
                if normalize_latents: x = pixel_norm(x, epsilon=pixelnorm_epsilon)
                with tf.variable_scope('Dense'):
                    x = dense(x, fmaps=nf(res-1)*16, gain=np.sqrt(2)/4, use_wscale=use_wscale) # override gain to match the original Theano implementation
                    x = tf.reshape(x, [-1, nf(res-1), 4, 4])
                    x = PN(act(apply_bias(x)))
                with tf.variable_scope('Conv'):
                    x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
            else: # 8x8 and up
                if fused_scale:
                    with tf.variable_scope('Conv0_up'):
                        x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                else:
                    x = upscale2d(x)
                    with tf.variable_scope('Conv0'):
                        x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                with tf.variable_scope('Conv1'):
                    x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
            return x

    def torgb(x, res): # res = 2..resolution_log2
        lod = resolution_log2 - res
        with tf.variable_scope('ToRGB_lod%d' % lod):
            return apply_bias(conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale))

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        x = block(combo_in, 2)
        images_out = torgb(x, 2)
        for res in range(3, resolution_log2 + 1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = torgb(x, res)
            images_out = upscale2d(images_out)
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out = lerp_clip(img, images_out, lod_in - lod)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(x, res, lod):
            #print('-------------res=',res,'----------lod=',lod)
            y = block(x, res)
            img = lambda: upscale2d(torgb(y, res), 2**lod)
            if res > 2: img = cset(img, (lod_in > lod), lambda: upscale2d(lerp(torgb(y, res), upscale2d(torgb(x, res - 1)), lod_in - lod), 2**lod))
            if lod > 0: img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))
            return img()
        images_out = grow(combo_in, 2, resolution_log2 - 2)
        
    assert images_out.dtype == tf.as_dtype(dtype)
    images_out = tf.identity(images_out, name='images_out')
    return images_out
#----------------------------------------------------------------------------
#G_dcgan
#----------------------------------------------------------------------------
# Generator network used in the paper.

def G_paper_DCGAN(
    latents_in,                         # First input: Latent vectors [minibatch, latent_size].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    is_training=False,
    num_channels        = 1,            # Number of output color channels. Overridden based on dataset.
    resolution          = 32,           # Output resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    latent_size         = None,         # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
    normalize_latents   = True,         # Normalize latent vectors before feeding them to the network?
    use_wscale          = True,         # Enable equalized learning rate?
    use_pixelnorm       = True,         # Enable pixelwise feature vector normalization?
    pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
    use_leakyrelu       = False,         # True = leaky ReLU, False = ReLU.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = True,         # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    **kwargs):                          # Ignore unrecognized keyword args.

    print('This is kwarg',is_training)
    training=is_training
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def PN(x): return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x
    print('fmap_base=',fmap_base,'latent size=',latent_size)
    if latent_size is None: latent_size = nf(0)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu if use_leakyrelu else tf.nn.relu

    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    combo_in = tf.cast(tf.concat([latents_in, labels_in], axis=1), dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    # Building blocks.
    def block(x, res): # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            #print('**********************','%dx%d' % (2**res, 2**res))
            if res == 2: # 4x4
                if normalize_latents: x = pixel_norm(x, epsilon=pixelnorm_epsilon)
                with tf.variable_scope('Dense'):
                    x = dense(x, fmaps=nf(res-1)*16, gain=np.sqrt(2)/4, use_wscale=use_wscale) # override gain to match the original Theano implementation
                    x = tf.reshape(x, [-1, nf(res-1), 4, 4])
                    x = (act(bn(apply_bias(x),training=training)))
                with tf.variable_scope('Conv'):
                    x = (act(bn(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)),training=training)))
            else: # 8x8 and up
                if fused_scale:
                    with tf.variable_scope('Conv0_up'):
                        x = (act(bn(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)),training=training)))
                else:
                    x = upscale2d(x)
                    with tf.variable_scope('Conv0'):
                        x = (act(bn(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)),training=training)))
                with tf.variable_scope('Conv1'):
                    x = (act(bn(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)),training=training)))
            return x

    def torgb(x, res): # res = 2..resolution_log2
        lod = resolution_log2 - res
        with tf.variable_scope('ToRGB_lod%d' % lod):
            return apply_bias(conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale))

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        x = block(combo_in, 2)
        images_out = torgb(x, 2)
        for res in range(3, resolution_log2 + 1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = torgb(x, res)
            images_out = upscale2d(images_out)
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out = lerp_clip(img, images_out, lod_in - lod)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(x, res, lod):
            #print('-------------res=',res,'----------lod=',lod)
            y = block(x, res)
            img = lambda: upscale2d(torgb(y, res), 2**lod)
            if res > 2: img = cset(img, (lod_in > lod), lambda: upscale2d(lerp(torgb(y, res), upscale2d(torgb(x, res - 1)), lod_in - lod), 2**lod))
            if lod > 0: img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))
            return img()
        images_out = grow(combo_in, 2, resolution_log2 - 2)

    assert images_out.dtype == tf.as_dtype(dtype)
    #images_out=tf.tan(images_out, name='last_tanh')
    images_out = tf.identity(images_out, name='images_out')
    return images_out
#----------------------------------------------------------------------------
#capsule Generator
#----------------------------------------------------------------------------
# Generator network used in the paper.

def G_paper_caps(
    latents_in,                         # First input: Latent vectors [minibatch, latent_size].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    num_channels        = 1,            # Number of output color channels. Overridden based on dataset.
    resolution          = 32,           # Output resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    latent_size         = None,         # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
    normalize_latents   = True,         # Normalize latent vectors before feeding them to the network?
    use_wscale          = True,         # Enable equalized learning rate?
    use_pixelnorm       = True,         # Enable pixelwise feature vector normalization?
    pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
    use_leakyrelu       = True,         # True = leaky ReLU, False = ReLU.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = True,         # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    **kwargs):                          # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def PN(x): return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x
    def PN_caps(x,caps_dim):return pixel_norm_caps(x,epsilon=pixelnorm_epsilon,caps_dim=caps_dim)
    if latent_size is None: latent_size = nf(0)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu if use_leakyrelu else tf.nn.relu

    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    combo_in = tf.cast(tf.concat([latents_in, labels_in], axis=1), dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)
    #caps_d=[0,0,32,16,16,16] #cifar10_config
    #caps_d=[0,0,32,16,8,4]#cifar10
    #caps_d =[0,0,32,16,8,4,2,2] #CelebA
    #caps_d =[0,0,64,32,16,8,8,4] #LSUN
    #caps_d =[0,0,64,64,64,32,32,16] # LSUN config1 
    #caps_d =[0,0,64,64,64,32,32,32] # LSUN config2
    
    
    #caps_d =[0,0,64,32,32,16,16,16] # CelebA config1 
    #caps_d =[0,0,64,32,32,32,16,16] # CelebA config2 
    
    
    caps_d=[0,0,64,64,64,64,32,32,32] #  LSUN_cat config1
    
    scale=True
    iter=True #True
    # # Building blocks_caps.
    def block(x, res): # res = 2..resolution_log2
        caps_dim=caps_d[res]
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res == 2: # 4x4
                if normalize_latents: x = pixel_norm(x, epsilon=pixelnorm_epsilon)
                with tf.variable_scope('Dense'):
                    x = dense(x, fmaps=nf(res-1)*16, gain=np.sqrt(2)/4, use_wscale=use_wscale) # override gain to match the original Theano implementation
                    x = tf.reshape(x, [-1, nf(res-1), 4, 4])
                    x = PN(act(apply_bias(x)))
                    #print('%dx%d' % (2**res, 2**res),'Gen x 0',x.get_shape())
                with tf.variable_scope('Conv'):
                    #x= apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=32,use_wscale=use_wscale,scale=True),caps_dim=32)
                    x = PN_caps((apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=caps_dim,use_wscale=use_wscale,scale=scale,iter=iter),caps_dim=caps_dim)),caps_dim=caps_dim)
                    ###x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))####org
                    #x = PN(act(apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=32,use_wscale=use_wscale,scale=True),caps_dim=32)))

                    #print('%dx%d' % (2**res, 2**res),'Gen x conv******** ',x.get_shape())
            else: # 8x8 and up
                if fused_scale:
                    with tf.variable_scope('Conv0_up'):
                        x = PN_caps((apply_bias_caps(upscale2d_conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=caps_dim,use_wscale=use_wscale,scale=scale,iter=iter),caps_dim=caps_dim)),caps_dim=caps_dim)
                        #x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))) ######org
                        #x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))

                        #print('%dx%d' % (2**res, 2**res),'Gen x upscale2d_conv2d_caps',x.get_shape())
                else:
                    x = upscale2d(x)
                    #print('%dx%d' % (2**res, 2**res),'Gen x upscale ',x.get_shape())
                    with tf.variable_scope('Conv0'):
                        x = PN_caps((apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=caps_dim,use_wscale=use_wscale,scale=scale,iter=iter),caps_dim=caps_dim)),caps_dim=caps_dim)
                        ######x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))) # org

                with tf.variable_scope('Conv1'):
                    x = PN_caps((apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=caps_dim,use_wscale=use_wscale,scale=scale,iter=iter),caps_dim=caps_dim)),caps_dim=caps_dim)
                    ####x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))) #caps
                    #print('%dx%d' % (2**res, 2**res),'Gen x Conv1 ',x.get_shape())

            #print('******',caps_dim)
            return x



    def torgb(x, res): # res = 2..resolution_log2
        lod = resolution_log2 - res
        with tf.variable_scope('ToRGB_lod%d' % lod):
            return apply_bias(conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale))

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        x = block(combo_in, 2)
        images_out = torgb(x, 2)
        for res in range(3, resolution_log2 + 1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = torgb(x, res)
            images_out = upscale2d(images_out)
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out = lerp_clip(img, images_out, lod_in - lod)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(x, res, lod):
            y = block(x, res)
            img = lambda: upscale2d(torgb(y, res), 2**lod)
            if res > 2: img = cset(img, (lod_in > lod), lambda: upscale2d(lerp(torgb(y, res), upscale2d(torgb(x, res - 1)), lod_in - lod), 2**lod))
            if lod > 0: img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))
            return img()
        images_out = grow(combo_in, 2, resolution_log2 - 2)

    assert images_out.dtype == tf.as_dtype(dtype)
    images_out = tf.identity(images_out, name='images_out')
    return images_out

#----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Discriminator network used in the paper.

def G_paper_caps_half(
    latents_in,                         # First input: Latent vectors [minibatch, latent_size].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    num_channels        = 1,            # Number of output color channels. Overridden based on dataset.
    resolution          = 32,           # Output resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    latent_size         = None,         # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
    normalize_latents   = True,         # Normalize latent vectors before feeding them to the network?
    use_wscale          = True,         # Enable equalized learning rate?
    use_pixelnorm       = True,         # Enable pixelwise feature vector normalization?
    pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
    use_leakyrelu       = True,         # True = leaky ReLU, False = ReLU.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = True,         # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    **kwargs):                          # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def PN(x): return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x
    def PN_caps(x,caps_dim):return pixel_norm_caps(x,epsilon=pixelnorm_epsilon,caps_dim=caps_dim)
    if latent_size is None: latent_size = nf(0)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu if use_leakyrelu else tf.nn.relu

    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    combo_in = tf.cast(tf.concat([latents_in, labels_in], axis=1), dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)
    #caps_d=[0,0,32,16,16,16] #cifar10_config1
    #caps_d=[0,0,64,64,32,32] #cifar10_config2
    #conv_caps_F=[False,False,True,True,True,True] #Cifar10_config2
    #caps_d=[0,0,64,64,32,0] #cifar10_config3
    #print('caps_d=',caps_d)
    #conv_caps_F=[False,False,True,True,True,False] #Cifar10_config3
    
    #caps_d=[0,0,16,16,16,16] #cifar10_config3
    
    #conv_caps_F=[False,False,True,True,True,True] #Cifar10_config3
    #caps_d=[0,0,8,16,32,64] #cifar10_config4
    #conv_caps_F=[False,False,True,True,True,True] #Cifar10_config4
    
    #caps_d=[0,0,32,16,8,4]#cifar10
    #caps_d =[0,0,32,16,8,4,2,2] #CelebA config1
    #caps_d =[0,0,64,32,16,8,8,4] #LSUN
    #caps_d =[0,0,64,64,64,32,32,16] # LSUN config1 
    #caps_d =[0,0,64,64,64,32,32,32] # LSUN config2
    #caps_d =[0,0,128,128,0,0,0,0,0] # LSUN config3 &celebA config2
    #caps_d= [0,0, 0 ,128,64,0,0,0,0] # LSUN config4 &clebA config3
    #caps_d= [0,0, 128 ,128,64,0,0,0,0] # LSUN config5& celebA config4
    
    #caps_d= [0,0, 128 ,128,128,64,64,0,0] # LSUN 256 bedroomconfig 1 horse and cat
    #conv_caps_F=[False,False,True,True,True,True,True,False,False] #Lsunbedroom 256 hores and cat
    
    caps_d= [0,0, 128 ,128,64,64,64,0,0] # LSUN 256 bedroomconfig 2
    conv_caps_F=[False,False,True,True,True,True,True,False,False] #Lsunbedroom 2 256 hors and cat
    
    #caps_d= [0,0, 0 ,0,0,64,32,0] #  celebA config5 /config6 lsun
    #conv_caps_F=[False,False,True,True,False,False,False,False] #Lsun_config3 &celebA config2
    #conv_caps_F=[False,False,False,True,True,False,False,False] #Lsun_config4& clelbA_config3
    #conv_caps_F=[False,False,True,True,True,False,False,False] #Lsun_config5 &clebA_config4
    #conv_caps_F=[False,False,False,False,False,True,True,False] #clebA_config5 &config6 lsun
    
    #caps_d =[0,0,64,32,32,16,16,16] # CelebA config1 
    #caps_d =[0,0,64,32,32,32,16,16] # CelebA config2 
    
    
    #caps_d=[0,0,64,64,64,64,32,32,32] #  LSUN_cat config1
    print('caps_d=',caps_d)
    scale=True
    iter=True #True
    # # Building blocks_caps.
    def block(x, res): # res = 2..resolution_log2
        caps_dim=caps_d[res]
        ccf=conv_caps_F[res]
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res == 2: # 4x4
                if normalize_latents: x = pixel_norm(x, epsilon=pixelnorm_epsilon)
                with tf.variable_scope('Dense'):
                    x = dense(x, fmaps=nf(res-1)*16, gain=np.sqrt(2)/4, use_wscale=use_wscale) # override gain to match the original Theano implementation
                    x = tf.reshape(x, [-1, nf(res-1), 4, 4])
                    x = PN(act(apply_bias(x)))
                    #print('%dx%d' % (2**res, 2**res),'Gen x 0',x.get_shape())
                with tf.variable_scope('Conv'):
                    #x= apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=32,use_wscale=use_wscale,scale=True),caps_dim=32)
                    if ccf:
                        x = PN_caps((apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=caps_dim,use_wscale=use_wscale,scale=scale,iter=iter),caps_dim=caps_dim)),caps_dim=caps_dim)
                    else:
                        x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))####org
                    #x = PN(act(apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=32,use_wscale=use_wscale,scale=True),caps_dim=32)))

                    #print('%dx%d' % (2**res, 2**res),'Gen x conv******** ',x.get_shape())
            else: # 8x8 and up
                if fused_scale:
                    with tf.variable_scope('Conv0_up'):
                        if ccf:
                            x = PN_caps((apply_bias_caps(upscale2d_conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=caps_dim,use_wscale=use_wscale,scale=scale,iter=iter),caps_dim=caps_dim)),caps_dim=caps_dim)
                        else:
                            x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))) ######org
                        #x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))

                        #print('%dx%d' % (2**res, 2**res),'Gen x upscale2d_conv2d_caps',x.get_shape())
                else:
                    x = upscale2d(x)
                    #print('%dx%d' % (2**res, 2**res),'Gen x upscale ',x.get_shape())
                    with tf.variable_scope('Conv0'):
                        if ccf:
                            x = PN_caps((apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=caps_dim,use_wscale=use_wscale,scale=scale,iter=iter),caps_dim=caps_dim)),caps_dim=caps_dim)
                        else:
                            x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))) # org

                with tf.variable_scope('Conv1'):
                    if ccf:
                        x = PN_caps((apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=caps_dim,use_wscale=use_wscale,scale=scale,iter=iter),caps_dim=caps_dim)),caps_dim=caps_dim)
                    else:
                        x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))) #org
                    #print('%dx%d' % (2**res, 2**res),'Gen x Conv1 ',x.get_shape())

            #print('******',caps_dim)
            return x



    def torgb(x, res): # res = 2..resolution_log2
        lod = resolution_log2 - res
        with tf.variable_scope('ToRGB_lod%d' % lod):
            return apply_bias(conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale))

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        x = block(combo_in, 2)
        images_out = torgb(x, 2)
        for res in range(3, resolution_log2 + 1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = torgb(x, res)
            images_out = upscale2d(images_out)
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out = lerp_clip(img, images_out, lod_in - lod)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(x, res, lod):
            y = block(x, res)
            img = lambda: upscale2d(torgb(y, res), 2**lod)
            if res > 2: img = cset(img, (lod_in > lod), lambda: upscale2d(lerp(torgb(y, res), upscale2d(torgb(x, res - 1)), lod_in - lod), 2**lod))
            if lod > 0: img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))
            return img()
        images_out = grow(combo_in, 2, resolution_log2 - 2)

    assert images_out.dtype == tf.as_dtype(dtype)
    images_out = tf.identity(images_out, name='images_out')
    return images_out

#----------------------------------------------------------------------------

def G_paper_caps_half_bi(
    latents_in,                         # First input: Latent vectors [minibatch, latent_size].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    num_channels        = 1,            # Number of output color channels. Overridden based on dataset.
    resolution          = 32,           # Output resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    latent_size         = None,         # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
    normalize_latents   = True,         # Normalize latent vectors before feeding them to the network?
    use_wscale          = True,         # Enable equalized learning rate?
    use_pixelnorm       = True,         # Enable pixelwise feature vector normalization?
    pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
    use_leakyrelu       = True,         # True = leaky ReLU, False = ReLU.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = True,         # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    **kwargs):                          # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def PN(x): return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x
    def PN_caps(x,caps_dim):return pixel_norm_caps(x,epsilon=pixelnorm_epsilon,caps_dim=caps_dim)
    if latent_size is None: latent_size = nf(0)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu if use_leakyrelu else tf.nn.relu

    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    combo_in = tf.cast(tf.concat([latents_in, labels_in], axis=1), dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)
    #caps_d=[0,0,32,16,16,16] #cifar10_config1
    #caps_d=[0,0,64,64,32,32] #cifar10_config2
    #conv_caps_F=[False,False,True,True,True,True] #Cifar10_config2
    #caps_d=[0,0,64,64,32,0] #cifar10_config3
    #print('caps_d=',caps_d)
    #conv_caps_F=[False,False,True,True,True,False] #Cifar10_config3
    
    #caps_d=[0,0,16,16,16,16] #cifar10_config3
    
    #conv_caps_F=[False,False,True,True,True,True] #Cifar10_config3
    #caps_d=[0,0,8,16,32,64] #cifar10_config4
    #conv_caps_F=[False,False,True,True,True,True] #Cifar10_config4
    
    #caps_d=[0,0,32,16,8,4]#cifar10
    #caps_d =[0,0,32,16,8,4,2,2] #CelebA config1
    #caps_d =[0,0,64,32,16,8,8,4] #LSUN
    #caps_d =[0,0,64,64,64,32,32,16] # LSUN config1 
    #caps_d =[0,0,64,64,64,32,32,32] # LSUN config2
    #caps_d =[0,0,128,128,0,0,0,0,0] # LSUN config3 &celebA config2
    #caps_d= [0,0, 0 ,128,64,0,0,0,0] # LSUN config4 &clebA config3
    #caps_d= [0,0, 128 ,128,64,0,0,0,0] # LSUN config5& celebA config4
    
    #caps_d= [0,0, 128 ,128,128,64,64,0,0] # LSUN 256 bedroomconfig 1 horse and cat
    #conv_caps_F=[False,False,True,True,True,True,True,False,False] #Lsunbedroom 256 hores and cat
    
    #caps_d= [0,0, 128 ,128,64,64,64,0,0] # LSUN 256 bedroomconfig 2
    #conv_caps_F=[False,False,True,True,True,True,True,False,False] #Lsunbedroom 2 256 hors and cat
    
    #caps_d= [0,0, 0 ,0,0,64,32,0] #  celebA config5 /config6 lsun
    #conv_caps_F=[False,False,True,True,False,False,False,False] #Lsun_config3 &celebA config2
    #conv_caps_F=[False,False,False,True,True,False,False,False] #Lsun_config4& clelbA_config3
    #conv_caps_F=[False,False,True,True,True,False,False,False] #Lsun_config5 &clebA_config4
    #conv_caps_F=[False,False,False,False,False,True,True,False] #clebA_config5 &config6 lsun
    
    #caps_d =[0,0,64,32,32,16,16,16] # CelebA config1 
    #caps_d =[0,0,64,32,32,32,16,16] # CelebA config2 
    
    
    caps_d=[0,0,64,64,64,64,32,32,32] #  LSUN_cat config1
    conv_caps_F=[False,False,True,True,True,True,True,True,True] #Lsunbedroom 2 256 hors and cat config1
    print('caps_d=',caps_d)
    scale=True
    iter=True #True
    # # Building blocks_caps.
    def block(x, res): # res = 2..resolution_log2
        caps_dim=caps_d[res]
        ccf=conv_caps_F[res]
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res == 2: # 4x4
                if normalize_latents: x = pixel_norm(x, epsilon=pixelnorm_epsilon)
                with tf.variable_scope('Dense'):
                    x = dense(x, fmaps=nf(res-1)*16, gain=np.sqrt(2)/4, use_wscale=use_wscale) # override gain to match the original Theano implementation
                    x = tf.reshape(x, [-1, nf(res-1), 4, 4])
                    x = PN(act(apply_bias(x)))
                    #print('%dx%d' % (2**res, 2**res),'Gen x 0',x.get_shape())
                with tf.variable_scope('Conv'):
                    #x= apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=32,use_wscale=use_wscale,scale=True),caps_dim=32)
                    if ccf:
                        x = PN_caps((apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=caps_dim,use_wscale=use_wscale,scale=scale,iter=iter),caps_dim=caps_dim)),caps_dim=caps_dim)
                    else:
                        x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))####org
                    #x = PN(act(apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=32,use_wscale=use_wscale,scale=True),caps_dim=32)))

                    #print('%dx%d' % (2**res, 2**res),'Gen x conv******** ',x.get_shape())
            else: # 8x8 and up
                if fused_scale:
                    with tf.variable_scope('Conv0_up'):
                        if ccf:
                            x = PN_caps((apply_bias_caps(upscale2d_conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=caps_dim,use_wscale=use_wscale,scale=scale,iter=iter),caps_dim=caps_dim)),caps_dim=caps_dim)
                        else:
                            x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))) ######org
                        #x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))

                        #print('%dx%d' % (2**res, 2**res),'Gen x upscale2d_conv2d_caps',x.get_shape())
                else:
                    #x = upscale2d(x)
                    x = bi_upscale2d(x)
                    #print('%dx%d' % (2**res, 2**res),'Gen x upscale ',x.get_shape())
                    with tf.variable_scope('Conv0'):
                        if ccf:
                            x = PN_caps((apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=caps_dim,use_wscale=use_wscale,scale=scale,iter=iter),caps_dim=caps_dim)),caps_dim=caps_dim)
                        else:
                            x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))) # org

                with tf.variable_scope('Conv1'):
                    if ccf:
                        x = PN_caps((apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=caps_dim,use_wscale=use_wscale,scale=scale,iter=iter),caps_dim=caps_dim)),caps_dim=caps_dim)
                    else:
                        x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))) #org
                    #print('%dx%d' % (2**res, 2**res),'Gen x Conv1 ',x.get_shape())

            #print('******',caps_dim)
            return x



    def torgb(x, res): # res = 2..resolution_log2
        lod = resolution_log2 - res
        with tf.variable_scope('ToRGB_lod%d' % lod):
            return apply_bias(conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale))

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        x = block(combo_in, 2)
        images_out = torgb(x, 2)
        for res in range(3, resolution_log2 + 1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = torgb(x, res)
            images_out = upscale2d(images_out)
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out = lerp_clip(img, images_out, lod_in - lod)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(x, res, lod):
            y = block(x, res)
            img = lambda: upscale2d(torgb(y, res), 2**lod)
            if res > 2: img = cset(img, (lod_in > lod), lambda: upscale2d(lerp(torgb(y, res), upscale2d(torgb(x, res - 1)), lod_in - lod), 2**lod))
            if lod > 0: img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))
            return img()
        images_out = grow(combo_in, 2, resolution_log2 - 2)

    assert images_out.dtype == tf.as_dtype(dtype)
    images_out = tf.identity(images_out, name='images_out')
    return images_out

#----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Discriminator network used in the paper.




#-----------------------------------------------------------------------------
# Generator  network with score activation function


def G_paper_caps_half_score(
    latents_in,                         # First input: Latent vectors [minibatch, latent_size].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    num_channels        = 1,            # Number of output color channels. Overridden based on dataset.
    resolution          = 32,           # Output resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    latent_size         = None,         # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
    normalize_latents   = True,         # Normalize latent vectors before feeding them to the network?
    use_wscale          = True,         # Enable equalized learning rate?
    use_pixelnorm       = True,         # Enable pixelwise feature vector normalization?
    pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
    use_leakyrelu       = True,         # True = leaky ReLU, False = ReLU.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = True,         # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    **kwargs):                          # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def PN(x): return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x
    def PN_caps(x,caps_dim):return pixel_norm_caps(x,epsilon=pixelnorm_epsilon,caps_dim=caps_dim)
    def SA(x,caps_dim,th): return score_act(x, epsilon=1e-7, caps_dim=caps_dim, th=th)
    if latent_size is None: latent_size = nf(0)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu if use_leakyrelu else tf.nn.relu

    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    combo_in = tf.cast(tf.concat([latents_in, labels_in], axis=1), dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)
    #caps_d=[0,0,32,16,16,16] #cifar10_config1
    #caps_d=[0,0,64,64,32,32] #cifar10_config2
    #conv_caps_F=[False,False,True,True,True,True] #Cifar10_config2
    #caps_d=[0,0,64,64,32,0] #cifar10_config3
    #print('caps_d=',caps_d)
    #conv_caps_F=[False,False,True,True,True,False] #Cifar10_config3
    
    #caps_d=[0,0,16,16,16,16] #cifar10_config3
    
    #conv_caps_F=[False,False,True,True,True,True] #Cifar10_config3
    #caps_d=[0,0,8,16,32,64] #cifar10_config4
    #conv_caps_F=[False,False,True,True,True,True] #Cifar10_config4
    
    #caps_d=[0,0,32,16,8,4]#cifar10
    #caps_d =[0,0,32,16,8,4,2,2] #CelebA config1
    #caps_d =[0,0,64,32,16,8,8,4] #LSUN
    #caps_d =[0,0,64,64,64,32,32,16] # LSUN config1 
    #caps_d =[0,0,64,64,64,32,32,32] # LSUN config2
    #caps_d =[0,0,128,128,0,0,0,0,0] # LSUN config3 &celebA config2
    #caps_d= [0,0, 0 ,128,64,0,0,0,0] # LSUN config4 &clebA config3
    #caps_d= [0,0, 128 ,128,64,0,0,0,0] # LSUN config5& celebA config4
    
    #caps_d= [0,0, 128 ,128,128,64,64,0,0] # LSUN 256 bedroomconfig 1 horse and cat
    #conv_caps_F=[False,False,True,True,True,True,True,False,False] #Lsunbedroom 256 hores and cat
    
    #caps_d= [0,0, 128 ,128,64,64,64,0,0] # LSUN 256 bedroomconfig 2
    #conv_caps_F=[False,False,True,True,True,True,True,False,False] #Lsunbedroom 2 256 hors and cat
    
    #caps_d= [0,0, 0 ,0,0,64,32,0] #  celebA config5 /config6 lsun
    #caps_d= [0,0, 0 ,0,128,64,32,0] #  celebA config6 /config7 lsun
    #caps_d= [0,0, 0 ,128,128,64,32,0] #  celebA config7 /config8 lsun
    caps_d= [0,0, 128 ,128,128,64,32,0] #  celebA config8 /config9 lsun
    #conv_caps_F=[False,False,True,True,False,False,False,False] #Lsun_config3 &celebA config2
    #conv_caps_F=[False,False,False,True,True,False,False,False] #Lsun_config4& clelbA_config3
    #conv_caps_F=[False,False,True,True,True,False,False,False] #Lsun_config5 &clebA_config4
    #conv_caps_F=[False,False,False,False,False,True,True,False] #clebA_config5 &config6 lsun
    #conv_caps_F=[False,False,False,False,True,True,True,False] #clebA_config6 &config7 lsun
    #conv_caps_F=[False,False,False,True,True,True,True,False] #clebA_config7 &config8 lsun
    conv_caps_F=[False,False,True,True,True,True,True,False] #clebA_config7 &config8 lsun
    
    #caps_d =[0,0,64,32,32,16,16,16] # CelebA config1 
    #caps_d =[0,0,64,32,32,32,16,16] # CelebA config2 
    
    
    #caps_d=[0,0,64,64,64,64,32,32,32] #  LSUN_cat config1
    
    thrshld=[0,0,0.3,0.3,0.3,0.3,0.3,0.3,0.3]
    print('caps_d=',caps_d)
    scale=True
    iter=True #True
    # # Building blocks_caps.
    def block(x, res): # res = 2..resolution_log2
        caps_dim=caps_d[res]
        ccf=conv_caps_F[res]
        th=thrshld[res]
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res == 2: # 4x4
                if normalize_latents: x = pixel_norm(x, epsilon=pixelnorm_epsilon)
                with tf.variable_scope('Dense'):
                    x = dense(x, fmaps=nf(res-1)*16, gain=np.sqrt(2)/4, use_wscale=use_wscale) # override gain to match the original Theano implementation
                    x = tf.reshape(x, [-1, nf(res-1), 4, 4])
                    x = PN(act(apply_bias(x)))
                    #print('%dx%d' % (2**res, 2**res),'Gen x 0',x.get_shape())
                with tf.variable_scope('Conv'):
                    #x= apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=32,use_wscale=use_wscale,scale=True),caps_dim=32)
                    if ccf:
                        x = SA((apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=caps_dim,use_wscale=use_wscale,scale=scale,iter=iter),caps_dim=caps_dim)),caps_dim=caps_dim,th=th)
                    else:
                        x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))####org
                    #x = PN(act(apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=32,use_wscale=use_wscale,scale=True),caps_dim=32)))

                    #print('%dx%d' % (2**res, 2**res),'Gen x conv******** ',x.get_shape())
            else: # 8x8 and up
                if fused_scale:
                    with tf.variable_scope('Conv0_up'):
                        if ccf:
                            x = SA((apply_bias_caps(upscale2d_conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=caps_dim,use_wscale=use_wscale,scale=scale,iter=iter),caps_dim=caps_dim)),caps_dim=caps_dim,th=th)
                        else:
                            x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))) ######org
                        #x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))

                        #print('%dx%d' % (2**res, 2**res),'Gen x upscale2d_conv2d_caps',x.get_shape())
                else:
                    x = upscale2d(x)
                    #print('%dx%d' % (2**res, 2**res),'Gen x upscale ',x.get_shape())
                    with tf.variable_scope('Conv0'):
                        if ccf:
                            x = SA((apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=caps_dim,use_wscale=use_wscale,scale=scale,iter=iter),caps_dim=caps_dim)),caps_dim=caps_dim,th=th)
                        else:
                            x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))) # org

                with tf.variable_scope('Conv1'):
                    if ccf:
                        x = SA((apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3, caps_dim=caps_dim,use_wscale=use_wscale,scale=scale,iter=iter),caps_dim=caps_dim)),caps_dim=caps_dim,th=th)
                    else:
                        x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))) #org
                    #print('%dx%d' % (2**res, 2**res),'Gen x Conv1 ',x.get_shape())

            #print('******',caps_dim)
            return x



    def torgb(x, res): # res = 2..resolution_log2
        lod = resolution_log2 - res
        with tf.variable_scope('ToRGB_lod%d' % lod):
            return apply_bias(conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale))

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        x = block(combo_in, 2)
        images_out = torgb(x, 2)
        for res in range(3, resolution_log2 + 1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = torgb(x, res)
            images_out = upscale2d(images_out)
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out = lerp_clip(img, images_out, lod_in - lod)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(x, res, lod):
            y = block(x, res)
            img = lambda: upscale2d(torgb(y, res), 2**lod)
            if res > 2: img = cset(img, (lod_in > lod), lambda: upscale2d(lerp(torgb(y, res), upscale2d(torgb(x, res - 1)), lod_in - lod), 2**lod))
            if lod > 0: img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))
            return img()
        images_out = grow(combo_in, 2, resolution_log2 - 2)

    assert images_out.dtype == tf.as_dtype(dtype)
    images_out = tf.identity(images_out, name='images_out')
    return images_out

#----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Discriminator network used in the paper.




def D_paper(
    images_in,                          # Input: Images [minibatch, channel, height, width].
    num_channels        = 1,            # Number of input color channels. Overridden based on dataset.
    resolution          = 32,           # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    use_wscale          = True,         # Enable equalized learning rate?
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = True,         # True = use fused conv2d + downscale2d, False = separate downscale2d layers.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    **kwargs):                          # Ignore unrecognized keyword args.
    
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    # Building blocks.
    def fromrgb(x, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
            return act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=1, use_wscale=use_wscale)))
    def block(x, res): # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res >= 3: # 8x8 and up
                with tf.variable_scope('Conv0'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                    #print('$$$$$$$$$$$$$$$$x_Conv0',x.get_shape())
                if fused_scale:
                    with tf.variable_scope('Conv1_down'):
                        x = act(apply_bias(conv2d_downscale2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                        #print('$$$$$$$$$$$$$$$$x_conv1_down',x.get_shape())
                else:
                    with tf.variable_scope('Conv1'):
                        x = act(apply_bias(conv2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                    x = downscale2d(x)
            else: # 4x4
                if mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, mbstd_group_size)
                with tf.variable_scope('Conv'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                with tf.variable_scope('Dense0'):
                    x = act(apply_bias(dense(x, fmaps=nf(res-2), use_wscale=use_wscale)))
                with tf.variable_scope('Dense1'):
                    x = apply_bias(dense(x, fmaps=1+label_size, gain=1, use_wscale=use_wscale))
            return x
    
    # Linear structure: simple but inefficient.
    if structure == 'linear':
        img = images_in
        x = fromrgb(img, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            lod = resolution_log2 - res
            x = block(x, res)
            #print('Disc&&&&&&&&&&&&&&&&&&&&&&',' Grow_lod%d' % lod,x.get_shape())
            img = downscale2d(img)
            y = fromrgb(img, res - 1)
            #print('DiscYYYYYYYYY&&&&&&&&&&&&&&&&&&&&&&',' Grow_lod%d' % lod,y.get_shape())
            with tf.variable_scope('Grow_lod%d' % lod):
                x = lerp_clip(x, y, lod_in - lod)
        combo_out = block(x, 2)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(res, lod):
            x = lambda: fromrgb(downscale2d(images_in, 2**lod), res)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = block(x(), res); y = lambda: x
            if res > 2: y = cset(y, (lod_in > lod), lambda: lerp(x, fromrgb(downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))
            return y()
        combo_out = grow(2, resolution_log2 - 2)

    assert combo_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(combo_out[:, :1], name='scores_out')
    labels_out = tf.identity(combo_out[:, 1:], name='labels_out')
    return scores_out, labels_out

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Discriminator_caps network used in the paper.

def D_paper_caps(
    images_in,                          # Input: Images [minibatch, channel, height, width].
    num_channels        = 1,            # Number of input color channels. Overridden based on dataset.
    resolution          = 32,           # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    use_wscale          = True,         # Enable equalized learning rate?
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = True,         # True = use fused conv2d + downscale2d, False = separate downscale2d layers.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
    **kwargs):                          # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def PN_caps(x,caps_dim):return pixel_norm_caps(x,epsilon=pixelnorm_epsilon,caps_dim=caps_dim)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)
    caps_dim=[0,0,16,16,8,8,4,4]
    # Building blocks.
    def fromrgb(x, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
            return act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=1, use_wscale=use_wscale)))
    def block(x, res): # res = 2..resolution_log2
        caps_d=caps_dim[res]
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res >= 3: # 8x8 and up
                with tf.variable_scope('Conv0'):
                    #x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                    x = PN_caps(apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3,caps_dim=caps_d, use_wscale=use_wscale,scale=True),caps_dim=caps_d),caps_dim=caps_d)
                    #print('$$$$$$$$$$$$$$$$x_Conv0',x.get_shape())
                if fused_scale:
                    with tf.variable_scope('Conv1_down'):
                        #x = act(apply_bias(conv2d_downscale2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                        #x=conv2d_downscale2d_caps(x, fmaps=nf(res-1), kernel=3,caps_dim=16, use_wscale=use_wscale,scale=True)
                        x = PN_caps(apply_bias_caps(conv2d_downscale2d_caps(x, fmaps=nf(res-2), kernel=3,caps_dim=caps_d, use_wscale=use_wscale,scale=True),caps_dim=caps_d),caps_dim=caps_d)
                        #print('$$$$$$$$$$$$$$$$x_conv1_down',x.get_shape())
                else:
                    with tf.variable_scope('Conv1'):
                        #x = act(apply_bias(conv2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                        x = PN_caps(apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3,caps_dim=caps_d, use_wscale=use_wscale,scale=True),caps_dim=caps_d),caps_dim=caps_d)
                    x = downscale2d(x)
            else: # 4x4
                if mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, mbstd_group_size)
                with tf.variable_scope('Conv'):
                    #x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                    x = PN_caps(apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3,caps_dim=caps_d, use_wscale=use_wscale,scale=True),caps_dim=caps_d),caps_dim=caps_d)
                with tf.variable_scope('Dense0'):
                    x = act(apply_bias(dense(x, fmaps=nf(res-2), use_wscale=use_wscale)))
                with tf.variable_scope('Dense1'):
                    x = apply_bias(dense(x, fmaps=1+label_size, gain=1, use_wscale=use_wscale))
            return x

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        img = images_in
        x = fromrgb(img, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            lod = resolution_log2 - res

            x = block(x, res)
            #print('&&&&&&&&&&&&&&&&&&&&&&',x.get_shape())
            img = downscale2d(img)
            y = fromrgb(img, res - 1)
            #print('YYYYYYYYY&&&&&&&&&&&&&&&&&&&&&&',y.get_shape())
            with tf.variable_scope('Grow_lod%d' % lod):
                x = lerp_clip(x, y, lod_in - lod)
        combo_out = block(x, 2)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(res, lod):
            x = lambda: fromrgb(downscale2d(images_in, 2**lod), res)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = block(x(), res); y = lambda: x
            if res > 2: y = cset(y, (lod_in > lod), lambda: lerp(x, fromrgb(downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))
            return y()
        combo_out = grow(2, resolution_log2 - 2)

    assert combo_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(combo_out[:, :1], name='scores_out')
    labels_out = tf.identity(combo_out[:, 1:], name='labels_out')
    return scores_out, labels_out

#----------------------------------------------------------------------------
#D_dcgan
#-----------------------------------------------------------------------------
# Discriminator network used in the paper.

def D_paper_DCGAN(
    images_in,                          # Input: Images [minibatch, channel, height, width].
    num_channels        = 1,            # Number of input color channels. Overridden based on dataset.
    resolution          = 32,           # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    use_wscale          = True,         # Enable equalized learning rate?
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = True,         # True = use fused conv2d + downscale2d, False = separate downscale2d layers.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    #training            = True,         # training for batch normalization
    **kwargs):                          # Ignore unrecognized keyword args.
    training=True
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    # Building blocks.
    def fromrgb(x, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
            return act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=1, use_wscale=use_wscale)))
    def block(x, res): # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res >= 3: # 8x8 and up
                with tf.variable_scope('Conv0'):
                    x = act(ln(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)),training=training))
                    #print('$$$$$$$$$$$$$$$$x_Conv0',x.get_shape())
                if fused_scale:
                    with tf.variable_scope('Conv1_down'):
                        x = act(ln(apply_bias(conv2d_downscale2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)),training=training))
                        #print('$$$$$$$$$$$$$$$$x_conv1_down',x.get_shape())
                else:
                    with tf.variable_scope('Conv1'):
                        x = act(ln(apply_bias(conv2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)),training=training))
                    x = downscale2d(x)
            else: # 4x4
                if mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, mbstd_group_size)
                with tf.variable_scope('Conv'):
                    x = act (ln(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)),training=training))
                with tf.variable_scope('Dense0'):
                    x = act(apply_bias(dense(x, fmaps=nf(res-2), use_wscale=use_wscale)))
                with tf.variable_scope('Dense1'):
                    x = apply_bias(dense(x, fmaps=1+label_size, gain=1, use_wscale=use_wscale))
            return x

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        img = images_in
        x = fromrgb(img, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            lod = resolution_log2 - res
            x = block(x, res)
            #print('Disc&&&&&&&&&&&&&&&&&&&&&&',' Grow_lod%d' % lod,x.get_shape())
            img = downscale2d(img)
            y = fromrgb(img, res - 1)
            #print('DiscYYYYYYYYY&&&&&&&&&&&&&&&&&&&&&&',' Grow_lod%d' % lod,y.get_shape())
            with tf.variable_scope('Grow_lod%d' % lod):
                x = lerp_clip(x, y, lod_in - lod)
        combo_out = block(x, 2)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(res, lod):
            x = lambda: fromrgb(downscale2d(images_in, 2**lod), res)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = block(x(), res); y = lambda: x
            if res > 2: y = cset(y, (lod_in > lod), lambda: lerp(x, fromrgb(downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))
            return y()
        combo_out = grow(2, resolution_log2 - 2)

    assert combo_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(combo_out[:, :1], name='scores_out')
    labels_out = tf.identity(combo_out[:, 1:], name='labels_out')
    return scores_out, labels_out

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
 #Discriminator capsconv_capsdense

def D_paper_capsconv_capsdense(
    images_in,                          # Input: Images [minibatch, channel, height, width].
    num_channels        = 1,            # Number of input color channels. Overridden based on dataset.
    resolution          = 32,           # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    use_wscale          = True,         # Enable equalized learning rate?
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = True,         # True = use fused conv2d + downscale2d, False = separate downscale2d layers.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
    **kwargs):                          # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def PN_caps(x,caps_dim):return pixel_norm_caps(x,epsilon=pixelnorm_epsilon,caps_dim=caps_dim)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)
    caps_dim=[0,0,16,16,8,4,2,2]
    # Building blocks.
    def fromrgb(x, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
            return act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=1, use_wscale=use_wscale)))
    def block(x, res): # res = 2..resolution_log2
        caps_d=caps_dim[res]
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res >= 3: # 8x8 and up
                with tf.variable_scope('Conv0'):
                    #x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                    x = (apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3,caps_dim=caps_d, use_wscale=use_wscale,scale=True),caps_dim=caps_d))
                if fused_scale:
                    with tf.variable_scope('Conv1_down'):
                        #x = act(apply_bias(conv2d_downscale2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                        x = PN_caps(apply_bias_caps(conv2d_downscale2d_caps(x, fmaps=nf(res-2), kernel=3,caps_dim=caps_d, use_wscale=use_wscale,scale=True),caps_dim=caps_d),caps_dim=caps_d)
                else:
                    with tf.variable_scope('Conv1'):
                        #x = act(apply_bias(conv2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                        x = PN_caps(apply_bias_caps(conv2d_caps(x, fmaps=nf(res-2), kernel=3,caps_dim=caps_d, use_wscale=use_wscale,scale=True),caps_dim=caps_d),caps_dim=caps_d)
                    x = downscale2d(x)
            else: # 4x4
                if mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, mbstd_group_size)
                with tf.variable_scope('Conv'):
                    #x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                    x = PN_caps(apply_bias_caps(conv2d_caps(x, fmaps=nf(res-1), kernel=3,caps_dim=caps_d, use_wscale=use_wscale,scale=True),caps_dim=caps_d),caps_dim=caps_d)
                with tf.variable_scope('Dense0'):
                    x=PN_caps(apply_bias_caps(conv_dense_caps(x,fmaps=nf(res-2), use_wscale=use_wscale,caps_dim=32),caps_dim=32),caps_dim=32)
                    #x = act(apply_bias(dense(x, fmaps=nf(res-2), use_wscale=use_wscale)))
                with tf.variable_scope('Dense1'):
                    x = apply_bias(dense(x, fmaps=1+label_size, gain=1, use_wscale=use_wscale))
            return x

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        img = images_in
        x = fromrgb(img, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = downscale2d(img)
            y = fromrgb(img, res - 1)
            with tf.variable_scope('Grow_lod%d' % lod):
                x = lerp_clip(x, y, lod_in - lod)
        combo_out = block(x, 2)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(res, lod):
            x = lambda: fromrgb(downscale2d(images_in, 2**lod), res)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = block(x(), res); y = lambda: x
            if res > 2: y = cset(y, (lod_in > lod), lambda: lerp(x, fromrgb(downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))
            return y()
        combo_out = grow(2, resolution_log2 - 2)

    assert combo_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(combo_out[:, :1], name='scores_out')
    labels_out = tf.identity(combo_out[:, 1:], name='labels_out')
    return scores_out, labels_out

#----------------------------------------------------------------------------

