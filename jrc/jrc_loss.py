def listwise_loss(y_true, y_pred, context_index):
  y_true = tf.cast(y_true, tf.float32)
  batch_size = tf.shape(y_true)[0]
  context_index = tf.cast(context_index, tf.float32)
  print('context_index:%s' % context_index)
  tf.debugging.check_numerics(context_index, 'context_index nan')
  #context_index = tf.ones([batch_size, 1], tf.float32)
  mask = tf.equal(context_index, tf.transpose(context_index))
  print('mask:%s' % mask)

  y_pred = tf.tile(tf.expand_dims(y_pred, 1), [1, batch_size, 1])
  y_true = tf.tile(tf.expand_dims(y_true, 1), [1, batch_size, 1])
  mask = tf.cast(mask, tf.float32)
  y_pred = y_pred + (1-tf.expand_dims(mask, 2)) * -1e9  #* float('-inf')
  y_true = y_true * tf.expand_dims(mask, 2)
  true_neg, true_pos = y_true[:,:,0], y_true[:,:,1]
  pred_neg, pred_pos = y_pred[:,:,0], y_pred[:,:,1]

  loss_pos = -K.sum(true_pos * tf.math.log(tf.nn.softmax(pred_pos, axis=0) + 1e-9), axis=0)
  loss_neg = -K.sum(true_neg * tf.math.log(tf.nn.softmax(pred_neg, axis=0) + 1e-9), axis=0)
  ge_loss = K.mean((loss_pos+loss_neg)/(K.sum(mask, axis=0)))
  return ge_loss

def two_logits_pointwise_listwise_loss(y_true, y_pred):
  #label[1] click state label[0] non-click state
  batch_size = tf.shape(y_true)[0]
  context_index = y_true[:,2:3]
  #context_index = tf.ones([batch_size, 1], tf.float32)
  weight = 0.9
  pointwise_loss = K.mean(K.categorical_crossentropy(y_true, y_pred, from_logits=True))
  ll = listwise_loss(y_true, y_pred, context_index)
  return weight * pointwise_loss + (1-weight) * ll
