import axesRSA.setup as s
from axesRSA.byRSAz import *
import kaoetal14.Kao1 as km
from scipy.stats import pearsonr

dfde = s.pd.read_csv('data/kao-F-ANT.csv')
# dfde = dfde.replace({'buffalo': 'bison', 'cow':'cattle'})
cf = s.config(sources='data/config/GLOVe/SOURCE.csv', intents='data/config/GLOVe/INTENTS.csv',
              axes='data/config/GLOVe/AXES.csv')
l = modCOMPLETE(cf)

# experimental_conditions = pd.read_json("validation data/Other Data/KaoQUD-Condition4Guides/KaoStimuli-4.json")
# experimental_conditions['question'] = [val.split()[-1].replace('?','') for val in experimental_conditions['question'].values]

experimental_conditions = pd.read_csv("data/Kao Original/KaoStimuli-4.csv")
experimental_conditions['question'] = [val.replace('?','') for val in experimental_conditions['question'].values]

kaoModConvert = pd.read_csv("data/Kao Original/kao-to-mod-dic.csv")

human_data = pd.read_csv("data/Kao Original/data39-long.csv")

def cond4_data():
    cond4 = []
    for cond in experimental_conditions[['animal', 'question', 'f1', 'f2', 'f3']].values:
        loc = dfde[['f', 'f-', 'composite.axis']].loc[dfde['animal'].isin([cond[0]]) & dfde['f'].isin([cond[1]])].values[0]
        allaxes = dfde['composite.axis'].loc[dfde['animal'].isin([cond[0]])].values
        convertVocab = dfde['f'].loc[dfde['animal'].isin([cond[0]])].values
        convertDic = {kaoModConvert['kao.f'].loc[kaoModConvert['animal'].isin([cond[0]]) & kaoModConvert['model.f'].isin([val])].values[0]:val for val in convertVocab}

        KaoG = {i: .2 for i in cond[2:]}
        KaoG[cond[1]] = .6

        kao_resp = km.LnG(cond[0],KaoG,np.array([.01,.99])).sum(axis=1)[1].tolist()

        pq = torch.zeros(size=(len(cf.ax),))
        pq[cf.sel_axis(allaxes)] = .1
        pq[cf.sel_axis([loc[-1]])] = .8

        pc = torch.FloatTensor([.01, .99])
        if cond[0] == 'tiger':
            pc = torch.FloatTensor([.99, .01])

        resp = l.lnn(cond[0],'human',pq=pq,pc=pc,lam=3)
        resp = [resp[cf.sel_intent([convertDic[i]]),1].item() for i in cond[2:] if i != 'quacking']
        if cond[0] == 'duck':
            resp.append(0.)
        cond4.append([cond[0], cond[1]]+resp+kao_resp)

    return pd.DataFrame(np.array(cond4), columns=['animal', 'QUD', 'mod.f1', 'mod.f2', 'mod.f3', 'kao.f1', 'kao.f2', 'kao.f3'])

def cond2_data():
    cond2 = []
    for cond in experimental_conditions[['animal', 'question', 'f1', 'f2', 'f3']].values:
        loc = dfde[['f', 'f-', 'composite.axis']].loc[dfde['animal'].isin([cond[0]]) & dfde['f'].isin([cond[1]])].values[0]
        allaxes = dfde['composite.axis'].loc[dfde['animal'].isin([cond[0]])].values
        convertVocab = dfde['f'].loc[dfde['animal'].isin([cond[0]])].values
        convertDic = {kaoModConvert['kao.f'].loc[kaoModConvert['animal'].isin([cond[0]]) & kaoModConvert['model.f'].isin([val])].values[0]:val for val in convertVocab}

        KaoG = {i: 1 for i in cond[2:]}

        kao_resp = km.LnG(cond[0],KaoG,np.array([.01,.99])).sum(axis=1)[1].tolist()

        pq = torch.zeros(size=(len(cf.ax),))
        pq[cf.sel_axis(allaxes)] = 1

        pc = torch.FloatTensor([.01, .99])
        if cond[0] == 'tiger':
            pc = torch.FloatTensor([.99, .01])

        resp = l.lnn(cond[0],'human',pq=pq,pc=pc,lam=3)
        resp = [resp[cf.sel_intent([convertDic[i]]),1].item() for i in cond[2:] if i != 'quacking']
        if cond[0] == 'duck':
            resp.append(0.)
        cond2.append([cond[0], 'uniform']+resp+kao_resp)

    return pd.DataFrame(np.array(cond2), columns=['animal', 'QUD', 'mod.f1', 'mod.f2', 'mod.f3', 'kao.f1', 'kao.f2', 'kao.f3'])

def rectfy_with_human(cond_data, condition):
    dfh = human_data[['animal', 'condition', 'workerid', 'f1prob', 'f2prob', 'f3prob']].loc[human_data['condition'].isin([condition])].copy()
    dfh.index=range(len(dfh))
    dfh.columns = ['animal', 'condition', 'id', 'hum.f1', 'hum.f2', 'hum.f3']

    data = np.array([cond_data[list(cond_data)[2:5]].loc[cond_data['animal'].isin([animal])].values[0] for animal in dfh['animal'].values])
    dfh[list(cond_data)[2:5]] = data

    return dfh

def analysis_setup(cond_data):
    data = []
    for row in cond_data.values:
        headers = [row[0], row[1]]
        outvals = list(zip(row[2:5], row[5:]))
        for i, val in enumerate(outvals):
            data.append(headers + [i] + [*val])
    return pd.DataFrame(np.array(data), columns=['animal', 'QUD', 'num', 'mod.f', 'kao.f'])

def analysis_setup_v2(cond_data):
    data = []
    for row in cond_data.values:
        headers = [row[0], row[1], row[2]]
        outvals = list(zip(row[3:6], row[6:]))
        for i, val in enumerate(outvals):
            data.append(headers + [i] + [*val])
    return pd.DataFrame(np.array(data), columns=['animal', 'QUD', 'id', 'num', 'col1.f', 'col2.f'])


def correlation_setup(conddata, columns1=['mod.f1','mod.f2','mod.f3'], columns2=['prob.f1','prob.f2','prob.f3']):
    data = []
    for idx in conddata.index:
        metadata = conddata[['animal', 'QUD']].loc[idx].values.tolist()
        model_values = conddata[columns1].loc[idx].values
        human_values = conddata[columns2].loc[idx].values
        for i, val in enumerate(list(zip(model_values, human_values))):
            data.append([metadata[0], i] + [*val])
    data = pd.DataFrame(np.array(data), columns=['animal', 'Fn', 'col1.p', 'col2.p'])
    data[['col1.p', 'col2.p']] = data[['col1.p', 'col2.p']].astype(np.float)
    return data

def global_correlation_data(df):
    global_cor = []
    for an in df['animal'].unique():
        ansub = df.loc[df['animal'].isin([an])]
        for num in ansub['num'].unique():
            ansubnum = ansub.loc[ansub['num'].isin([num])]
            global_cor.append([ansubnum['col1.f'].values.mean(), ansubnum['col2.f'].values.mean()])
    return np.array(global_cor)
