import matplotlib.pyplot as plt
import io

def _render_chart_image(chart_type: str, title: str, series_data: list, ordered_cats: list) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if chart_type == 'pieChart' and len(series_data) == 1:
        vals = [float(v) if v else 0 for v in series_data[0]['vals']]
        ax.pie(vals, labels=ordered_cats, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
    elif chart_type == 'lineChart':
        for s in series_data:
            vals = [float(v) if v else 0 for v in s['vals']]
            ax.plot(ordered_cats, vals, marker='o', label=s['name'])
        if len(series_data) > 1 or series_data[0]['name'] != 'Series':
            ax.legend()
    else:
        # Default to bar chart
        import numpy as np
        x = np.arange(len(ordered_cats))
        width = 0.8 / len(series_data)
        for i, s in enumerate(series_data):
            vals = [float(v) if v else 0 for v in s['vals']]
            ax.bar(x + i*width - 0.4 + width/2, vals, width, label=s['name'])
        ax.set_xticks(x)
        ax.set_xticklabels(ordered_cats)
        if len(series_data) > 1 or series_data[0]['name'] != 'Series':
            ax.legend()
            
    if title:
        ax.set_title(title)
        
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    return buf.getvalue()

series_data = [{'name': 'Percentage', 'cats': ['Computer', 'English'], 'vals': ['0.25', '0.2']}]
buf = _render_chart_image('pieChart', 'Test Pie', series_data, ['Computer', 'English'])
print(f"Generated {len(buf)} bytes")
