
import plotly.graph_objects as go


data = [['Kp20k', 0.346-0.298, 3.29], ['Inspec',0.525-0.333, 7.68], ['Krapivin',0.307-0.285, 3.20], ['NUS',0.369-0.366, 5.89], ['Semeval', 0.355-0.352, 6.71], ['KPTimes', 0.469-0.424, 2.40], ['JPTimes', 0.360-0.238, 3.86], ['DUC', 0.355-0.063, 7.79]]
#data = [['Kp20k', 0.346-0.298, 0.85], ['Inspec',0.525-0.333, 0.48], ['Krapivin',0.307-0.285, 0.76], ['NUS',0.369-0.366, 0.78], ['Semeval', 0.355-0.352, 0.63], ['KPTimes', 0.469-0.424, 0.95], ['JPTimes', 0.360-0.238, 0.61], ['DUC', 0.355-0.063, 0.17]]

x_y = sorted(data, key=lambda x:x[2])

y = [i[1] for i in x_y]
x = [i[2] for i in x_y]
text = [i[0] for i in x_y]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x,
    y=y,
    name="",
    mode='lines+markers+text',
    text=text,
    textposition='top center',
    marker=dict(size=12)
))


fig.update_layout(
    #xaxis_title= "Percentage of test set keywords that appear in train set",
    xaxis_title= "Avg. present keywords per document",
    yaxis_title= "Difference in F@10",
    font=dict(
        size=24)
)
fig.show()










