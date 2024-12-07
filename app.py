from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from markdown import markdown
import subprocess
import requests
import openai
import json
import os
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import plotly.io as pio
from scipy.spatial.distance import cosine
import plotly.graph_objects as go
from scipy.signal import find_peaks
from scipy.stats import boxcox
from scipy.interpolate import interp1d
app = Flask(__name__)
openai.api_key = "123"  # 替换为你的 OpenAI API 密钥
app.secret_key = 'your_secret_key_here'
# 已知的化合物列表
COMPOUND_NAMES = [
                "Gender", "Age", "Adipic Acid", "cis-Aconitic Acid", "Isocitric Acid",
                "α-Ketoglutaric Acid", "Malic Acid", "Succinic Acid", 
                "3-Hydroxy-3-Methylglutaric Acid", "α-Ketoisovaleric Acid", 
                "α-Ketoisocaproic Acid", "α-Keto-β-Methylvaleric Acid", 
                "Xanthurenic Acid", "Suberic Acid", "β-Hydroxyisovaleric Acid", 
                "3-Hydroxypropionic Acid", "Glutaric Acid", "Isovalerylglycine", 
                "α-Ketoadipic Acid", "Methylmalonic Acid", "Vanillylmandelic Acid", 
                "Homovanillic Acid", "5-Hydroxyindoleacetic Acid", "Kynurenic Acid", 
                "Sebacic Acid", "Quinolinic Acid", "Picolinic Acid", 
                "3-Methoxy-4-Hydroxyphenylethylene Glycol", "Mandelic Acid", 
                "Hydroxyphenylacetic Acid", "L-Pyroglutamic Acid", 
                "α-Hydroxyisobutyric Acid", "Benzoic Acid", "Hippuric Acid", 
                "Phenylacetic Acid", "Ethylmalonic Acid", "4-Hydroxybenzoic Acid", 
                "4-Hydroxyphenylacetic Acid", "3-Indoleacetic Acid", 
                "3,4-Dihydroxyphenylpropionic Acid", "Citramalic Acid", "Tartaric Acid", 
                "3-Hydroxyphenylacetic Acid", "p-Cresol", 
                "5-Hydroxymethyl-2-Furancarboxylic Acid", "2-Hydroxyphenylacetic Acid", 
                "Methylsuccinic Acid", "Homogentisic Acid", "3-Methylglutaconic Acid", 
                "3-Methylglutaric Acid", "3-Hydroxyglutaric Acid", 
                "N-Acetylaspartic Acid", "4-Hydroxyphenyllactic Acid", 
                "Phosphoric Acid", "Glyceric Acid", "Glycolic Acid", "Oxalic Acid", 
                "Pyruvic Acid", "Lactic Acid", "3-Hydroxybutyric Acid", "Citric Acid"
]

@app.route('/')
def index():
    return render_template('index.html', compound_names=COMPOUND_NAMES)

@app.route('/top3', methods=['GET', 'POST'])
def top3():
    if request.method == 'POST':
        patient_id = request.form.get('patient_id')
        result = subprocess.run(['./patient_match_top3', patient_id], capture_output=True, text=True)
        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError:
            return jsonify({"error": "Unable to parse output"}), 500
        return jsonify(output)
    return render_template('top3.html')

@app.route('/range', methods=['GET', 'POST'])
def range_match():
    if request.method == 'POST':
        patient_id = request.form.get('patient_id')
        lower_bound = request.form.get('lower_bound')
        upper_bound = request.form.get('upper_bound')
        result = subprocess.run(['./patient_match_range', patient_id, lower_bound, upper_bound], capture_output=True, text=True)
        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError:
            return jsonify({"error": "Unable to parse output"}), 500
        return jsonify(output)
    return render_template('range.html')

@app.route('/kmeans_results')
def kmeans_results():
    result_file_path = "kmeans_results.txt"
    if os.path.exists(result_file_path):
        clusters = []
        evaluations = []
        with open(result_file_path, 'r') as file:
            lines = file.readlines()
            cluster_started = False
            for line in lines:
                if line.startswith("Cluster"):
                    cluster_started = True
                    clusters.append(line.strip())
                elif cluster_started and line.strip():
                    clusters[-1] += "\n" + line.strip()
                elif line.startswith("Cluster Evaluation"):
                    cluster_started = False
                    evaluations.append(line.strip())
                elif not cluster_started and line.strip():
                    evaluations.append(line.strip())
        return render_template('kmeans_results.html', clusters=clusters, evaluations=evaluations)
    else:
        return "Results not found", 404
@app.route('/plot', methods=['GET', 'POST'])

def plot_data():
    
    graph_json_original = None
    graph_json_transformed = None
    outliers_original = None
    outliers_transformed = None
    user_value_difference = None
    if request.method == 'POST':
        file_path = request.form.get('file_path')
        user_value = request.form.get('user_value', type=float)
        if not os.path.exists(file_path):
            return "File not found. Please check the path.", 404

        try:
            data = pd.read_csv(file_path)
            # 确保所有列正确转换为数值类型
            for column in data.columns[1:]:  # 假设第一列不是数值数据
                data[column] = pd.to_numeric(data[column], errors='coerce')
        except Exception as e:
            return f"Error reading CSV file: {e}", 500

        # 读取过滤条件
        filters = {}
        for compound in COMPOUND_NAMES:
            min_val = request.form.get(f'{compound}_min', type=float)
            max_val = request.form.get(f'{compound}_max', type=float)
            if min_val is not None and max_val is not None:
                filters[compound] = (min_val, max_val)

        # 应用过滤条件
        for compound, (min_val, max_val) in filters.items():
            column_index = COMPOUND_NAMES.index(compound) + 1  # +1 assumes first column is ID or similar
            data = data[(data.iloc[:, column_index] >= min_val) & (data.iloc[:, column_index] <= max_val)]

        # 获取用户选择的目标列
        selected_column_name = request.form.get('column_name')
        if selected_column_name not in COMPOUND_NAMES:
            return "Invalid column selected. Please try again.", 400

        column_index = COMPOUND_NAMES.index(selected_column_name) + 1
        indicator_data = data.iloc[:, column_index].dropna()

        if not indicator_data.empty:
            # 生成原始图像
            fig_original, outliers_original, mpv,upper_bound = plot_original_distribution(indicator_data, selected_column_name)
            graph_json_original = pio.to_json(fig_original)
            if user_value is not None:
                # 计算用户输入值与 MPV 的差距
                #_, mpv = get_mpv(indicator_data)  # 假设 get_mpv 是计算并返回 MPV 的函数
                user_value_difference = 100*(user_value - mpv) / (upper_bound - mpv)
                print(f"User Value Difference: {upper_bound}")  # 打印到控制台
            # 生成 Box-Cox 变换图像
            fig_transformed, outliers_transformed = plot_transformed_distribution(indicator_data, selected_column_name)
            graph_json_transformed = pio.to_json(fig_transformed)
        else:
            return "No data available after applying filters.", 404

    return render_template('plot.html', compound_names=COMPOUND_NAMES,
                           graph_json_original=graph_json_original, graph_json_transformed=graph_json_transformed,
                           outliers_original=outliers_original, outliers_transformed=outliers_transformed,user_value_difference=user_value_difference)

    # Step 1: 计算 MPV（模式值）

def plot_original_distribution(indicator_data, indicator_name):

   # Step 1: 计算 MPV（模式值）
    hist, bin_edges = np.histogram(indicator_data, bins=20, density=True)
    max_index = np.argmax(hist)  # 找到直方图最高点的索引
    mpv = (bin_edges[max_index] + bin_edges[max_index + 1]) / 2  # 聚众值（模式值）

    # Step 2: 获取柱状图顶点的横坐标和高度
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # 每个柱状图的中心点
    heights = hist  # 每个柱状图的高度

    # Step 3: 插值生成平滑曲线
    interp_func = interp1d(bin_centers, heights, kind='cubic', fill_value="extrapolate")
    extended_x = np.linspace(min(bin_centers), max(bin_centers), 2000)
    smoothed_y = interp_func(extended_x)

    # Step 4: 检测异常值
    threshold_98 = np.percentile(indicator_data, 98)
    outliers = [(index, value) for index, value in indicator_data.items() if value > threshold_98]

    # Step 5: 绘制图表
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Histogram(
        x=indicator_data, histnorm='probability density', name='原始数据图像', opacity=0.6
    ))

    fig.add_trace(go.Scatter(
        x=extended_x, y=smoothed_y, mode='lines', name='Smoothed Curve'
    ), secondary_y=True)

    #fig.add_trace(go.Scatter(
     #   x=[value for _, value in outliers], y=[0] * len(outliers),
      #  mode='markers', marker=dict(color='red', size=4), name='Outliers'
    #))

    # Step 6: 添加注释 - 输出 MPV 和 ±2σ
    lower_bound = mpv - 2 * np.std(indicator_data)
    upper_bound = mpv + 2 * np.std(indicator_data)

    fig.add_annotation(
        text=f"\u200b\u200b\n\nMPV = {mpv:.2f}, -2σ = {lower_bound:.2f}, +2σ = {upper_bound:.2f}",
        xref="paper", yref="paper",
        x=0.5, y=-0.2, showarrow=False,
        font=dict(size=26),  # 调整字体大小
        align="center"
    )


    # 更新布局
    fig.update_layout(
        title=f'{indicator_name} Distribution (Original Data)',
        xaxis_title='',
        yaxis_title='Density'
    )

    return fig, outliers, mpv, upper_bound
def chain_function(indicator_data, indicator_name):
    with open("output_log.txt", "a") as log_file:  # 以追加模式打开文件
        log_file.write(f"Indicator Name: {indicator_name}\n")
        fig, outliers, mpv, upper_bound = plot_original_distribution(indicator_data, indicator_name)
        plot_transformed_distribution(mpv, upper_bound)
        log_file.write(f"MPV (Most Probable Value): {mpv}\n")
        log_file.write(f"Upper Bound: {upper_bound}\n")
    return mpv


def chain_function1(indicator_data, indicator_name):
    fig, outliers, mpv, upper_bound = plot_original_distribution(indicator_data, indicator_name)
    plot_transformed_distribution(mpv, upper_bound)
    return upper_bound


def plot_transformed_distribution(indicator_data, indicator_name):
    # Step 1: Box-Cox 变换
    
    try:
        indicator_data_transformed, _ = boxcox(indicator_data + 1e-9)  # 防止 0 值
    except ValueError:
        return None, None  # 如果无法进行 Box-Cox 变换，直接返回

    # Step 2: 正态分布拟合
    mu, std = norm.fit(indicator_data_transformed)

    # Step 3: 扩展 x 轴范围
# Step 3: 扩展 x 轴范围并加上偏移量
    #offset = 2.0  # 偏移量，可以是正数（右移）或负数（左移）
    offset = chain_function(indicator_data, indicator_name)

    std = (chain_function1(indicator_data, indicator_name) - offset)/2
    extended_x = np.linspace(min(indicator_data_transformed) - 3 * std,
                              max(indicator_data_transformed) + 3 * std, 200)
    p = norm.pdf(extended_x, mu, std)

    # Step 4: 检测异常值
    threshold_98 = norm.ppf(1, mu, std)
    outliers = [(index, value) for index, value in indicator_data.items() if value > threshold_98]

    lower_bound1 = mu - 3 * std
    upper_bound1 = mu + 3 * std

    indicator_data_transformed_scaled = np.interp(
    indicator_data_transformed,  # 输入数据
    (indicator_data_transformed.min(), indicator_data_transformed.max()),  # 原始范围
    (lower_bound1, upper_bound1)  # 目标范围
)
    # Step 5: 绘制图表
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Histogram(
        x=indicator_data , histnorm='probability density', name='Original Data Distribution', opacity=0.6
    ))
#_transformed_scaled - mu + offset
    fig.add_trace(go.Scatter(
        x=extended_x  - mu + offset, y=p, mode='lines', name=f'Normal Distribution'
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=[value for _, value in outliers], y=[0] * len(outliers),
        mode='markers', marker=dict(color='red', size=8), name='Outliers'
    ))

    # Step 6: 添加注释
    fig.add_annotation(
        text=f"Transformed Data: μ={mu:.2f}, σ={std:.2f}, Outliers Detected: {len(outliers)}",
        xref="paper", yref="paper",
        x=0.5, y=-0.2, showarrow=False,
        font=dict(size=1),
        align="center"
    )
    fig.add_annotation(
        text=f"\u200b\u200b\n\nMPV = {offset:.2f}, -2σ = {-2*std+offset:.2f}, +2σ = {2*std+offset:.2f}",
        xref="paper", yref="paper",
        x=0.5, y=-0.2, showarrow=False,
        font=dict(size=1),  # 调整字体大小
        align="center"
    )
    # 更新布局
    fig.update_layout(
        title=f'{indicator_name} Distribution (Box-Cox Transformed)',
        xaxis_title='Transformed Value',
        yaxis_title='Density'
    )

    return fig, outliers

@app.route('/input_data', methods=['GET', 'POST'])
def input_data():
    if request.method == 'POST':
        # 优先检查是否有文件名和病人索引的输入
        csv_file_path = request.form.get('file')  # 获取用户选择的文件路径
        patient_index_str = request.form.get('patient_index')  # 获取病人索引（字符串形式）

        if csv_file_path and patient_index_str:
            try:
                patient_index = int(patient_index_str)  # 尝试将病人索引转换为整数
            except ValueError:
                return f"Invalid patient index: {patient_index_str}.", 400

            if not os.path.isfile(csv_file_path):
                return f"File {csv_file_path} not found.", 404

            try:
                # 从文件中读取数据
                file_data = pd.read_csv(csv_file_path)
                if patient_index < 0 or patient_index >= len(file_data):
                    return f"Invalid patient index: {patient_index}.", 400

                # 提取指定病人的数据作为输入，并忽略第一列
                patient_data = file_data.iloc[patient_index, 1:].fillna(0).values.tolist()
                # 在控制台上打印读取的数据
                print(f"Data for patient index {patient_index} from file '{csv_file_path}': {patient_data}")
            except Exception as e:
                return f"Error reading CSV file: {e}", 500
        else:
            # 如果没有文件输入，从表单中获取用户输入
            patient_data = [request.form.get(compound, type=float) or 0 for compound in COMPOUND_NAMES]

            # 确保列数与 COMPOUND_NAMES 一致
            if len(patient_data) != len(COMPOUND_NAMES):
                return f"Data length mismatch. Expected {len(COMPOUND_NAMES)} columns, got {len(patient_data)}.", 400

        # 创建 DataFrame
        new_data_df = pd.DataFrame([patient_data], columns=COMPOUND_NAMES)

        # 示例处理文件路径
        test_csv_path = 'working/test.csv'
        csv_1234_path = 'working/1234.csv'

        # 处理 test.csv 文件
        if os.path.isfile(test_csv_path):
            existing_data_df = pd.read_csv(test_csv_path)
            combined_data_df = pd.concat([new_data_df, existing_data_df], ignore_index=True)

            # 如果不是从文件读取数据，则从表单中获取用户输入
            patient_data = [request.form.get(compound, type=float) or 0 for compound in COMPOUND_NAMES]

        new_data_df = pd.DataFrame([patient_data], columns=COMPOUND_NAMES)


        # 文件路径
        test_csv_path = 'working/test.csv'
        csv_1234_path = 'working/1234.csv'

        # 处理 test.csv 文件
        if os.path.isfile(test_csv_path):
            existing_data_df = pd.read_csv(test_csv_path)
            # 将新数据添加到现有数据的前面
            combined_data_df = pd.concat([new_data_df, existing_data_df], ignore_index=True)
        else:
            combined_data_df = new_data_df
        # 写回数据到 test.csv
        combined_data_df.to_csv(test_csv_path, index=False)

        # 处理 1234.csv 文件
        if os.path.isfile(csv_1234_path):
            existing_data_1234 = pd.read_csv(csv_1234_path)
        else:
            existing_data_1234 = pd.DataFrame()

        # 在新数据中添加id列
        new_data_df['id'] = 10001 + len(existing_data_1234)  # 开始的id根据现有行数决定
        new_data_df['id'] += new_data_df.index  # 每行增加index值，实现连续id

        # 将新数据追加到现有数据的末尾
        combined_data_1234 = pd.concat([existing_data_1234, new_data_df], ignore_index=True)

        # 写回数据到 1234.csv
        combined_data_1234.to_csv(csv_1234_path, index=False)
       

        # 读取 1234.csv 文件
        file_path = '1234 (copy).csv'
        if not os.path.exists(file_path):
            return "File not found. Please check the path.", 404

        try:
            data = pd.read_csv(file_path)
            # 确保所有数据为数值类型，并将空值替换为0
            for column in data.columns[1:]:  # 假设第一列不是数值数据
                data[column] = pd.to_numeric(data[column], errors='coerce').fillna(0)
        except Exception as e:
            return f"Error reading CSV file: {e}", 500

        # 确保 patient_data 和 data 的维度匹配
        if len(patient_data) != len(data.columns) - 1:
            return "Input data does not match the expected number of features.", 400


        # 计算每个指标的最聚众值
        crowding_values = []
        max_values = []
        for column in data.columns[1:]:
            bins = np.arange(data[column].min(), data[column].max() + 0.5, 0.5)
            counts, _ = np.histogram(data[column].dropna(), bins=bins)
            max_bin_index = np.argmax(counts)
            crowding_value = (bins[max_bin_index] + bins[max_bin_index + 1]) / 2
            crowding_values.append(crowding_value)
            max_values.append(data[column].max())

        # 计算每个指标的距离百分比和相对位置（高于或低于最聚众值）
        distances = []
        outliers = []
        for i, value in enumerate(patient_data):
            if i < len(crowding_values):
                crowding_value = crowding_values[i]
                max_value = max_values[i]
                if value <= crowding_value:
                    percentage = round((( crowding_value - value ) / (max_value - crowding_value)) * 100, 2)
                    distances.append({"status": "低于", "percentage": percentage})
                else:
                    percentage = round(((value - crowding_value) / (max_value - crowding_value)) * 100, 2)
                    distances.append({"status": "高于", "percentage": percentage})
                    # 如果值位于聚众值与最大值差的98%以上，判定为异常值
                    if percentage >= 98:
                        outliers.append({"index": i, "compound_name": COMPOUND_NAMES[i], "value": value})
            else:
                distances.append(None)

        # 计算余弦相似度和相似人群数量
        patient_vector = np.array(patient_data)
        data_vectors = data.iloc[:, 1:].dropna().values
        if patient_vector.shape[0] != data_vectors.shape[1]:
            return "Data dimensions do not match for similarity calculation.", 400

        similarities = [1 - cosine(patient_vector, vector) for vector in data_vectors]
        p = max(similarities)
        q = sum(1 for sim in similarities if sim >= p)

        # 计算相似度高于50%、60%、70%、80%、90%的病人人数
        similar_people_above_90 = sum(1 for sim in similarities if sim * 100 >= 90)
        similar_people_above_80 = sum(1 for sim in similarities if sim * 100 >= 80)
        similar_people_above_70 = sum(1 for sim in similarities if sim * 100 >= 70)
        similar_people_above_60 = sum(1 for sim in similarities if sim * 100 >= 60)
        similar_people_above_50 = sum(1 for sim in similarities if sim * 100 >= 50)
        similar_people_above_40 = sum(1 for sim in similarities if sim * 100 >= 40)
        similar_people_above_30 = sum(1 for sim in similarities if sim * 100 >= 30)
        similar_people_above_20 = sum(1 for sim in similarities if sim * 100 >= 20)
        similar_people_above_10 = sum(1 for sim in similarities if sim * 100 >= 10)
        similar_people_above_95 = sum(1 for sim in similarities if sim * 100 >= 95)
        # 找到最相似的病人数据
        most_similar_index = similarities.index(p)
        most_similar_patient = data.iloc[most_similar_index, 1:].to_dict()

        # 生成报告
        report = {
            "outlier_count": len(outliers),
            "outliers": outliers,
            "distances": distances,
            "crowding_values": crowding_values,
            "max_values": max_values,
            "similarity_percentage": round(p * 100, 2),
            "similar_people_count": q,
            "most_similar_patient": most_similar_patient,
            "compound_names": COMPOUND_NAMES,
            "patient_data": patient_data,
            "similar_people_above_90": similar_people_above_90,
            "similar_people_above_80": similar_people_above_80,
            "similar_people_above_70": similar_people_above_70,
            "similar_people_above_60": similar_people_above_60,
            "similar_people_above_50": similar_people_above_50,
            "similar_people_above_40": similar_people_above_40,
            "similar_people_above_30": similar_people_above_30,
            "similar_people_above_20": similar_people_above_20,
            "similar_people_above_10": similar_people_above_10,
            "similar_people_above_95": similar_people_above_95
        }
        session['report'] = report
        return render_template('report.html',
                               outlier_count=report['outlier_count'],
                               outliers=report['outliers'],
                               distances=report['distances'],
                               crowding_values=report['crowding_values'],
                               max_values=report['max_values'],
                               similarity_percentage=report['similarity_percentage'],
                               similar_people_count=report['similar_people_count'],
                               most_similar_patient=report['most_similar_patient'],
                               compound_names=report['compound_names'],
                               patient_data=report['patient_data'],
                               similar_people_above_90=report['similar_people_above_90'],
                               similar_people_above_80=report['similar_people_above_80'],
                               similar_people_above_70=report['similar_people_above_70'],
                               similar_people_above_60=report['similar_people_above_60'],
                               similar_people_above_50=report['similar_people_above_50'], similar_people_above_40=report['similar_people_above_40'],
                               similar_people_above_30=report['similar_people_above_30'],
                               similar_people_above_20=report['similar_people_above_20'],
                               similar_people_above_10=report['similar_people_above_10'],
                               similar_people_above_95=report['similar_people_above_95'])

    return render_template('input_data.html', compound_names=COMPOUND_NAMES)

@app.route('/chatgpt_report', methods=['GET'])
def chatgpt_report():
    # 从 session 中获取报告数据
    print("Attempting to retrieve report data from session.")
    report_data = session.get('report')

    # 如果 session 中没有 report_data，返回错误信息
    if not report_data:
        print("No report data available in session.")
        return jsonify({"error": "No report data available"}), 400

    try:
        # 构建 prompt 内容
        print("Building prompt with the report data.")
        prompt = f"""
        生成一份基于以下患者数据的详细报告：
        1. 异常指标数量：{report_data['outlier_count']}
        2. 异常指标列表：
        {", ".join([f"{outlier['compound_name']}: {outlier['value']}" for outlier in report_data['outliers']])}
        3. 相似度百分比：{report_data['similarity_percentage']}%
        4. 最相似的病人数据：
        {", ".join([f"{key}: {value}" for key, value in report_data['most_similar_patient'].items()])}
        5. 各指标详细数据：
        {", ".join([f"{compound}: {value}" for compound, value in zip(report_data['compound_names'], report_data['patient_data'])])}

        请根据上述数据生成一份结构化的文字报告，包含总结和详细分析部分。请多写一些文字分析，提供建议用药与潜在病症的分析。不要换行，txt文本格式
        """
        print("Generated Prompt:", prompt)

        # 请求 Node.js 服务来获取 ChatGPT 生成的内容
        node_js_api_url = "http://localhost:5001"
        print(f"Sending POST request to Node.js server at {node_js_api_url} with prompt.")

        response = requests.post(node_js_api_url, json={"prompt": prompt})
        print(f"Received response from Node.js server with status code: {response.status_code}")

        # 检查响应是否成功
        if response.status_code != 200:
            print("Error: Failed to get a successful response from Node.js server.")
            return jsonify({"error": "Failed to get response from Node.js server"}), response.status_code

        # 获取返回的报告文本
        report_text = response.json().get("message", "No content returned")
        print("Generated Report Text:", report_text)

        # 检查 report_text 是否是字典
        if isinstance(report_text, dict):
            print("Converting report_text from dict to string.")
            report_text = json.dumps(report_text, ensure_ascii=False, indent=4)

        # 生成 HTML 内容
        html_content = markdown(report_text)

        # 将报告写入到 md 文件，删除 '\' 和 'n'
        file_path = "patient_report.txt"
        print(f"Writing report to file: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as report_file:
            report_text_no_backslash_n = report_text.replace('\\', '').replace('n', '')
            report_file.write(report_text_no_backslash_n)
        print(f"Report successfully written to {file_path}")

        # 返回 HTML 响应展示报告内容
        html_response = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Patient Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                p {{ margin-bottom: 1em; }}
                ul, ol {{ padding-left: 20px; }}
                blockquote {{ border-left: 5px solid #ccc; padding-left: 10px; color: #666; }}
                pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; }}
                code {{ background-color: #eef; padding: 2px 4px; border-radius: 3px; }}
                br {{ margin-bottom: 1em; }}
            </style>
        </head>
        <body>
            <h1>患者详细报告</h1>
            <div id="markdown-content">
                {html_content}
            </div>
        </body>
        </html>
        """
        return html_response

    except requests.RequestException as e:
        print(f"Request to Node.js server failed: {str(e)}")
        return jsonify({"error": f"Request to Node.js server failed: {str(e)}"}), 500
    except Exception as e:
        print(f"Unexpected error occurred: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
@app.route('/report')
def report():
    report_data = session.get('report')
    if report_data:
        return render_template('report.html', data=report_data)
    else:
        return "报告数据不可用", 404

    return redirect(url_for('chatgpt_report'))
@app.route('/mpv')
def display_file():
    try:
        # 读取 Excel 文件
        file_path = "MPV DATA.xlsx"  # 指定 Excel 文件路径
        df = pd.read_excel(file_path)

        # 将数据转换为 HTML 表格
        table_html = df.to_html(classes='table table-bordered table-hover table-responsive', index=False)

        # 返回包含美观表格的网页
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>MPV Data</title>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <style>
                body {{
                    background-color: #f8f9fa;
                    font-family: Arial, sans-serif;
                }}
                .container {{
                    margin-top: 50px;
                    max-width: 90%;
                    padding: 20px;
                    background-color: #ffffff;
                    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                }}
                h1 {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                table {{
                    word-wrap: break-word;
                    table-layout: auto;
                    width: 100%;
                }}
                th, td {{
                    text-align: left;
                    padding: 8px;
                }}
                th {{
                    background-color: #007bff;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>MPV Data</h1>
                <div class="table-responsive">
                    {table_html}
                </div>
            </div>
        </body>
        </html>
        '''
    except Exception as e:
        return f"Error processing file: {str(e)}"
if __name__ == '__main__':
    app.run(debug=True)

