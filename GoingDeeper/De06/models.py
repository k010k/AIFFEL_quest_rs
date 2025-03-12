import numpy as np
import tensorflow as tf

# Positional Encoding 구현
def positional_encoding(pos, d_model):
    def cal_angle(position, i):
        return position / np.power(10000, (2*(i//2)) / np.float32(d_model))

    def get_posi_angle_vec(position):
        return [cal_angle(position, i) for i in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(pos)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return sinusoid_table

# Mask  생성하기
def generate_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def generate_lookahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def generate_masks(src, tgt):
    enc_mask = generate_padding_mask(src)
    dec_enc_mask = generate_padding_mask(src)

    dec_lookahead_mask = generate_lookahead_mask(tgt.shape[1])
    dec_tgt_padding_mask = generate_padding_mask(tgt)
    dec_mask = tf.maximum(dec_tgt_padding_mask, dec_lookahead_mask)

    return enc_mask, dec_enc_mask, dec_mask

# Multi Head Attention 구현
class MultiHeadAttention(tf.keras.layers.Layer):
    #1.attention score
    #2.attention coeffient
    #3.attention value
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.depth = d_model // self.num_heads
        
        self.W_q = tf.keras.layers.Dense(d_model)
        self.W_k = tf.keras.layers.Dense(d_model)
        self.W_v = tf.keras.layers.Dense(d_model)
        
        self.linear = tf.keras.layers.Dense(d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask):
        # TODO: 구현
        #0. d_k
        d_k = tf.cast(K.shape[-1], tf.float32)
        #1. dot product(Q,K)        
        QK = tf.matmul(Q,K, transpose_b=True)
        #2. scale(QK/self.d_model K)--> attetion score
        scaled_qk = QK / tf.math.sqrt(d_k)

        if mask is not None: scaled_qk += (mask * -1e9)
        #3. V sum(attetion coeffinetn,value)
        attentions = tf.nn.softmax(scaled_qk, axis=-1)
        out = tf.matmul(attentions, V)
        return out, attentions
        
    def split_heads(self, x):
        """
        입력 텐서를 여러 헤드로 나누는 함수
        
        Args:
            x: 입력 텐서 (shape: [batch_size, seq_len, d_model])
            num_heads: 헤드의 수
            
        Returns:
            분리된 텐서 (shape: [batch_size, num_heads, seq_len, depth])
        """
        batch_size = tf.shape(x)[0]

        # 텐서 reshaping(헤드 차원 추가)
        split_x = tf.reshape(x, [batch_size, -1, self.num_heads, self.depth])

        # 헤드 차원을 앞쪽으로 이동
        split_x = tf.transpose(split_x, perm=[0, 2, 1, 3])
        
        return split_x

    def combine_heads(self, x):
        """
        다수의 헤드로 분리된 텐서를 원래 상태로 병합하는 함수.
        
        Args:
            x: 분리된 텐서 (shape: [batch_size, num_heads, seq_len, depth])
            
        Returns:
            병합된 텐서 (shape: [batch_size, seq_len, d_model])
        """
        batch_size = tf.shape(x)[0]

        # 헤드 차원을 seq_len 차원 뒤로 이동(transpose)(
        combined_x = tf.transpose(x, perm=[0, 2, 1, 3])

        # 헤드 차원(num_heads, depth)을 합치기 (reshape)
        combined_x = tf.reshape(combined_x, (batch_size, -1, self.d_model))
        
        return combined_x

    def call(self, Q, K, V, mask):
        # TODO: 구현
        # 1. Q, K, V 선형 변환
        WQ = self.W_q(Q) # [batch_size, seq_len_q, d_model]
        WK = self.W_k(K) # [batch_size, seq_len_k, d_model]
        WV = self.W_v(V) # [batch_size, seq_len_v, d_model]

        # 2. 헤드 분리
        WQ_splits = self.split_heads(WQ) # [batch_size, num_heads, seq_leng_q, depth]
        WK_splits = self.split_heads(WK) # [batch_size, num_heads, seq_leng_k, depth]
        WV_splits = self.split_heads(WV) # [batch_size, num_heads, seq_leng_v, depth]        

        # 3. Scaled Dot-Product Attention 계산
        out, attention_weights = self.scaled_dot_product_attention(WQ_splits, WK_splits, WV_splits, mask)

        # 4. 헤드 병합
        out = self.combine_heads(out) # [batch_size, seq_len_q, d_model]

        #5. 최종 출력 생성
        out = self.linear(out)
        
        return out, attention_weights

# Position-wise Feed Forward Network 구현
class PoswiseFeedForwardNet(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.fc1 = tf.keras.layers.Dense(d_ff, activation='relu')
        self.fc2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        # 1. 첫 번째 레이어 (확장 및 활성화)
        out = self.fc1(x) # [batch_size, seq_len, d_ff]

        # 2. 두 번째 레이어 (축소)
        out = self.fc2(out) #[batch_size, seq_len, d_model]
        
        return out

# Encoder의 레이어 구현
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)

        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.do = tf.keras.layers.Dropout(dropout)
        
    def call(self, x, mask):
        '''
        Multi-Head Attention
        '''
        # TODO:  구현
        # 1. Multi-Head Attention + Residual Connection
        residual = x
        out = self.norm_1(x)# Normalization
        out, enc_attn = self.enc_self_attn(out, out, out, mask) # Self-Attention
        out = self.do(out) # Dropout
        out += residual
        '''
        Position-Wise Feed Forward Network
        '''
        # TODO: 구현
        # 2. Postion-wise Feed Forward Network
        residual = out
        out = self.norm_2(out)
        out = self.ffn(out)
        out = self.do(out)
        out += residual
        
        return out, enc_attn

# Decoder 레이어 구현
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()

        self.dec_self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)

        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)

        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.do = tf.keras.layers.Dropout(dropout)
    
    def call(self, x, enc_out, dec_enc_mask, padding_mask):

        '''
        Masked Multi-Head Attention
        '''
        # TODO: 구현
        residual = x
        out = self.norm_1(x)
        out, dec_attn = self.dec_self_attn(out, out, out, padding_mask)
        out = self.do(out)
        out += residual
        '''
        Multi-Head Attention
        '''
        # TODO: 구현
        residual = out
        out = self.norm_2(out)
        out, dec_enc_attn = self.enc_dec_attn(Q=out, K=enc_out, V=enc_out, mask=dec_enc_mask)
        out = self.do(out)
        out += residual
        '''
        Position-Wise Feed Forward Network
        '''
        # TODO: 구현
        residual = out
        out = self.norm_3(out)
        out = self.ffn(out)
        out = self.do(out)
        out += residual
        return out, dec_attn, dec_enc_attn

# Encoder 구현
class Encoder(tf.keras.Model):
    def __init__(self,
                    n_layers,
                    d_model,
                    n_heads,
                    d_ff,
                    dropout):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.enc_layers = [EncoderLayer(d_model, n_heads, d_ff, dropout) 
                        for _ in range(n_layers)]
    
        self.do = tf.keras.layers.Dropout(dropout)
        
    def call(self, x, mask):
        # TODO: 구현
        # 드롭아웃 적용 (첫 입력에만 사용)
        x = self.do(x)

        # Attention 가중치를 저장할 리스트
        enc_attns = []

        # 각 Encoder Layer를 순차적으로 실행
        for layer in self.enc_layers:
            x, attn = layer(x, mask) # Layer 호출
            enc_attns.append(attn)   # Attention 가중치 저장

        # 최종 출력 및 Attention 가중치 반환
        out = x

        return out, enc_attns

# Decoder 구현
class Decoder(tf.keras.Model):
    def __init__(self,
                    n_layers,
                    d_model,
                    n_heads,
                    d_ff,
                    dropout):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.dec_layers = [DecoderLayer(d_model, n_heads, d_ff, dropout) 
                            for _ in range(n_layers)]
                            
                            
    def call(self, x, enc_out, dec_enc_mask, padding_mask):
        # TODO: 구현
        #초기 입력
        out = x

        # 각 레이어의 Attention 가중치를 저장할 리스트
        dec_attns = []
        dec_enc_attns = []

        #모든 Decoder Layer를 순차적으로 실행
        for layer in self.dec_layers:
            # 각 레이어의 출력 및 Attention 가중치 계산
            out, dec_attn, dec_enc_attn = layer(out, enc_out, dec_enc_mask, padding_mask)

            # 가중치 저장
            dec_attns.append(dec_attn)
            dec_enc_attns.append(dec_enc_attn)
            
        return out, dec_attns, dec_enc_attns

class Transformer(tf.keras.Model):
    def __init__(self,
                    n_layers,
                    d_model,
                    n_heads,
                    d_ff,
                    src_vocab_size,
                    tgt_vocab_size,
                    pos_len,
                    dropout=0.2,
                    shared_fc=True,
                    shared_emb=False):
        super(Transformer, self).__init__()
        
        self.d_model = tf.cast(d_model, tf.float32)

        if shared_emb:
            self.enc_emb = self.dec_emb = \
            tf.keras.layers.Embedding(src_vocab_size, d_model)
        else:
            self.enc_emb = tf.keras.layers.Embedding(src_vocab_size, d_model)
            self.dec_emb = tf.keras.layers.Embedding(tgt_vocab_size, d_model)

        self.pos_encoding = positional_encoding(pos_len, d_model)
        self.do = tf.keras.layers.Dropout(dropout)

        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)

        self.fc = tf.keras.layers.Dense(tgt_vocab_size)

        self.shared_fc = shared_fc

        if shared_fc:
            self.fc.set_weights(tf.transpose(self.dec_emb.weights))

    def embedding(self, emb, x):
        # TODO: 구현
        seq_len = tf.shape(x)[1]

#         # Embedding lookup and scaling
#         out = emb(x) * tf.math.sqrt(self.d_model)

#         # Adding positional encoding
#         out += self.pos_encoding[:seq_len, :]

#         # Applying dropout
#         out = self.do(out)
        out = emb(x)

        if self.shared_fc: out *= tf.math.sqrt(self.d_model)

        out += self.pos_encoding[np.newaxis, ...][:, :seq_len, :]
        out = self.do(out)
        
        return out

    def call(self, enc_in, dec_in, enc_mask, dec_enc_mask, dec_mask):
        # TODO: 구현
        # Encoder embedding
        enc_embedded = self.embedding(self.enc_emb, enc_in) # [batch_size, input_seq_len, d_model]

        # Encoder forward pass
        enc_out, enc_attns = self.encoder(enc_embedded, enc_mask)

        # Decoder embedding
        dec_embedded = self.embedding(self.dec_emb, dec_in) # [batch_size, target_seq_len, d_model]
        
        # Decoder forward pass
        dec_out, dec_attns, dec_enc_attns = self.decoder(dec_embedded, enc_out, dec_enc_mask, dec_mask)

        # Final fully connected layer (projection to target vocab size)
        logits = self.fc(dec_out) # [batch_size, target_seq_len, tgt_vocab_size]
        
        return logits, enc_attns, dec_attns, dec_enc_attns