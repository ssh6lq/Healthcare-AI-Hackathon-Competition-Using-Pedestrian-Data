from sklearn.decomposition import PCA

pca = PCA(n_components = 1)

pca_x = pd.DataFrame(data = pca.fit_transform(df[["ax","gx"]]), columns=['x'])
print("x",pca.explained_variance_ratio_)
pca_y = pd.DataFrame(data = pca.fit_transform(df[["ay","gy"]]), columns=['y'])
print("y",pca.explained_variance_ratio_)
pca_z = pd.DataFrame(data = pca.fit_transform(df[["az","gz"]]), columns=['z'])
print("z",pca.explained_variance_ratio_)

df_pca = pd.concat([pca_x,pca_y,pca_z],axis=1)