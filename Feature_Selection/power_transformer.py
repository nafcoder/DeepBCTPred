from sklearn.preprocessing import PowerTransformer
import numpy as np
import pickle


train = np.load('features.npy')
new = np.load('features_top_images.npy')
test = np.load('features_test.npy')
combined = np.concatenate((train, new, test), axis=0)
print(combined.shape)

pt = PowerTransformer()

# Fit and transform the data
pt.fit(combined)

# save pt with pickle
with open('power_transformer.pkl', 'wb') as f:
    pickle.dump(pt, f)

# save transformed data
train_transformed = pt.transform(train)
new_transformed = pt.transform(new)
test_transformed = pt.transform(test)

np.save('features.npy', train_transformed)
np.save('features_top_images.npy', new_transformed)
np.save('features_test.npy', test_transformed)
