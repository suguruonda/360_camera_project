import pandas as pd

import plotly.express as px 

df = pd.read_csv('tax.csv')
df.loc[0,'Parent'] = ""
fig2 = px.sunburst(df, names='Item and Group',parents = 'Parent', values='Weight', color='Parent',branchvalues="total")
fig2.update_layout(title_text="", font_size=14)
fig2.write_image("sunburst.png")
#fig2.show()