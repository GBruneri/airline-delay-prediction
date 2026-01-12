from sklearn.pipeline import Pipeline


def train_pipeline(preprocessor, model, X_train, y_train):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline
