from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from surprise import Dataset
from surprise import Reader
from surprise import dump

rec_df = pd.read_csv('KNN_final.csv',index_col=0)
preds, model = dump.load('KNNfinal_model')
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(rec_df,reader)
trainset = data.build_full_trainset()
model.fit(trainset)

plot_df = pd.read_csv('with_plots.csv',index_col=0)

lst = plot_df['title']
app = Flask(__name__)

@app.route('/',methods=['GET', 'POST'])
@app.route('/home',methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return(render_template('home.html',Recommendations='',Titles=lst))
    elif request.method == 'POST':
        result = request.form
        for k,v in result.items():
            if k == 'Input_Text':
                ident = v
                break
        inner_id = model.trainset.to_inner_iid(ident)
        neighbors = model.get_neighbors(inner_id, k=5)
        raw = [model.trainset.to_raw_iid(iid)
                       for iid in neighbors]
        df = plot_df
        ids = list(df['title'])
        plots = list(df['Plot'])
        plot_list = list(zip(ids,plots))

        new_list = []
        for i in plot_list:
            if i[0] in raw:
                new_list.append(i)
        return(render_template('home.html',Recommendations=new_list,Titles=lst, select=ident))


if __name__ == '__main__':
    app.run(debug=True)
