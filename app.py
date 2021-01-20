import geopy.distance
import pandas as pd
from flask import Flask, jsonify
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

app = Flask(__name__)

ratings_list = pd.read_csv("ratings.csv")
movies_list = pd.read_csv("movies.csv")
supplier_list = pd.read_csv("suppliers.csv", engine="python")
detailsSupplier_list = pd.read_csv("details_suppliers.csv", engine="python")

ratings_df = pd.DataFrame(ratings_list)
movies_df = pd.DataFrame(movies_list)
suppliers_df = pd.DataFrame(supplier_list)
detailsSupplier_df = pd.DataFrame(detailsSupplier_list)

print(ratings_df.shape)


s_df = suppliers_df.set_index("idSupplier")


# specify rating scale


def check_recommended(user, item):
    recommended_list = pd.read_csv("recommended.csv")
    recommended_df = pd.DataFrame(recommended_list)
    return ((recommended_df["movieId"] == item) & (recommended_df["userId"] == user)).any()


def add_recommended(u, i):
    rec = {"userId": [u], "movieId": [i]}
    df = pd.DataFrame(rec)
    df.to_csv("recommended.csv", mode="a", header=False, index=False)


def create_model():
    model = SVD(n_factors=5, n_epochs=30, biased=False, lr_all=0.006)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df, reader)
    cross_validate(model, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
    return model


myModel = create_model()


def get_distance(lat1, lat2, long1, long2):
    coords_1 = (lat1, long1)
    coords_2 = (lat2, long2 )
    print(geopy.distance.great_circle(coords_1, coords_2).km)
    return geopy.distance.great_circle(coords_1, coords_2).km


def get_nearby_stores(current_lat, current_long):
    nearby_stores = []
    for id, lat, long, d in zip(suppliers_df["idSupplier"], suppliers_df["lat"], suppliers_df["long"],
                                suppliers_df["distanceRec"]):
        if (get_distance(lat, current_lat, long, current_long) <= d):
            nearby_stores.append(id)
    return nearby_stores


def get_best_products_of_nearby_stores(user, current_lat, current_long):
    maxItems = 0
    model = myModel
    products_of_nearby_stores = []
    nearby_stores = get_nearby_stores(current_lat, current_long)
    for x, y in zip(detailsSupplier_df["movieId"], detailsSupplier_df["idSupplier"]):
        if y in nearby_stores and model.predict(int(user), x).est >= 4.5 and (
                check_recommended(int(user), int(x)) == False):
            products_of_nearby_stores.append(x)
            maxItems += 1
        if maxItems == 10:
            return products_of_nearby_stores

    return products_of_nearby_stores


@app.route('/recommend/<idUser>/<lat>/<long>')
def get_recommendation(idUser, lat, long):
    products = get_best_products_of_nearby_stores(idUser, float(lat), float(long))
    # items and their suppliers
    items = []
    joinItem = movies_df.set_index("movieId").join(detailsSupplier_df.set_index("movieId"))

    for j in products[:]:
        itemSupplierId = joinItem.loc[j].idSupplier
        obj = {"title": joinItem.loc[j].title,
               "store": s_df.loc[itemSupplierId].store

               }
        items.append(obj)
        add_recommended(idUser, j)
    return jsonify(items)


if __name__ == '__main__':
    app.run(threaded=True)
