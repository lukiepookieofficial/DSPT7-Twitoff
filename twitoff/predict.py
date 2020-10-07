'''Predicts authorship of tweet based on embeddings.'''
import numpy as np
from sklearn.linear_model import LogisticRegression
from .db_model import User
from .twitter import nlp, vectorize_tweet


def predict_user(user1, user2, tweet_text):
    """Determine and return which user is more likely to say a given tweet.

    Args:
        user1 (str): Twitter user name for user 1 in comparison from web form
        user2 (str): Twitter user name for user 1 in comparison from web form
        tweet_text (str): Tweet text to evaluate from web form

    Returns:
        [type]: Prediction from LogReg model
    """

    user1 = User.query.filter(User.username == user1).one()
    user2 = User.query.filter(User.username == user2).one()
    user1_embeds = np.array([tweet.embedding for tweet in user1.tweet])
    user2_embeds = np.array([tweet.embedding for tweet in user2.tweet])

    # combine embeddings and create labels
    embeddings = np.vstack([user1_embeds, user2_embeds])
    labels = np.concatenate([np.ones(len(user1_embeds)),
                             np.zeroes(len(user2_embeds))])

    # train model and convert input text into embeddings
    log_reg = LogisticRegression(max_iter=1000).fir(embeddings, labels)
    tweet_embedding = vectorize_tweet(nlp, tweet_text)

    return log_reg.predict([tweet_embedding])[0]
