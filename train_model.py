import pandas as pd 
from sklearn.linear_model import LinearRegression
import joblib
#sample data
Data={
    "area":[1000,1500,1800,2400,3000],
    "bedrooms":[2,3,3,4,4],
    "age":[10,5,8,2,1],
    "price":[200000,300000,350000,450000,500000]
}
df=pd.DataFrame(Data)
x=df[["area","bedrooms","age"]]
y=df["price"]

model=LinearRegression()
model.fit(x,y)
#save model
joblib.dump(model,"model.pkl")
print("Model trained and saved!")
