# aicryptorisk by Travis Michael O'Dell 2025
import ipywidgets as widgets
from IPython.display import display, clear_output
import time, re, os, io, zipfile, requests, json, difflib
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Global settings dictionary (can be updated via the Settings tab)
global_settings = {
    "API_KEY": "YOUR_GEMINI_API_KEY",
    "API_ENGINE": "Gemini"  # Options: "GPT-4", "Gemini", "grok2"
}

# ================================
# Gemini Generative AI Integration
# ================================
class GeminiAI:
    @staticmethod
    def generate_analysis(prompt, context=None):
        """
        Uses the selected API engine to produce analysis.
        If the engine is Gemini, it calls Gemini's API.
        Otherwise, it simulates a response.
        """
        engine = global_settings.get("API_ENGINE", "Gemini")
        if engine == "Gemini":
            api_key = global_settings.get("API_KEY", "YOUR_GEMINI_API_KEY")
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            parts = [{"text": prompt}]
            if context:
                parts.append({"text": str(context)})
            payload = {"contents": [{"parts": parts}]}
            headers = {"Content-Type": "application/json"}
            try:
                response = requests.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                candidate = data.get("candidates", [{}])[0].get("output", "No output")
                return candidate
            except Exception as e:
                return f"[Gemini Analysis] {prompt} (simulated, error: {e})"
        else:
            return f"[{engine} Simulation] {prompt} with context {context}"

    @staticmethod
    def adapt_module(module_logic, input_data):
        print(f"[Gemini Adaptation] Adjusting module based on: {input_data}")
        return module_logic

# ================================
# Utility: Parse parameters from source text
# ================================
def parse_parameters(source_text):
    params = {
        "alpha": 0.5,
        "beta": 0.1,
        "gamma": 0.05,
        "U_max": 1.0,
        "safety_threshold": 10.0,
        "num_nodes": 20,
        "dt": 0.1,
        "T": 20.0
    }
    for line in source_text.splitlines():
        match = re.match(r'\s*(\w+)\s*=\s*([\d\.]+)', line)
        if match:
            key = match.group(1).lower()
            try:
                value = float(match.group(2))
            except:
                continue
            if key in params:
                params[key] = value
    return params

# ================================
# Module: Build and Initialize Network
# ================================
def build_network(num_nodes):
    G = nx.erdos_renyi_graph(n=int(num_nodes), p=0.3)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n=int(num_nodes), p=0.3)
    for i in G.nodes():
        G.nodes[i]['risk'] = np.random.uniform(0, 0.1)
        G.nodes[i]['region'] = np.random.choice([0, 1])
        G.nodes[i]['tech'] = np.random.uniform(0, 1)
        G.nodes[i]['policy'] = np.random.uniform(0, 1)
    return G

# ================================
# Module: External Forcing Function (4D Input)
# ================================
def external_forcing(t, node_data, U_max):
    r = node_data.get('region', 0)
    tech = node_data.get('tech', 0.5)
    policy = node_data.get('policy', 0.5)
    phase = r * np.pi
    amplitude = 1 + tech - policy
    return U_max * amplitude * np.sin(t + phase)

# ================================
# Module: Risk Dynamics Simulation (Euler integration)
# ================================
def simulate_risk_dynamics(G, params):
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    U_max = params["U_max"]
    dt = params["dt"]
    T = params["T"]
    num_steps = int(T/dt)
    
    risk_history = {node: [] for node in G.nodes()}
    total_risk_history = []
    
    for step in range(num_steps):
        t = step * dt
        current_risks = {node: G.nodes[node]['risk'] for node in G.nodes()}
        for i in G.nodes():
            dR = -alpha * current_risks[i]
            for j in G.neighbors(i):
                dR += beta * (current_risks[j] - current_risks[i])
            U_i = external_forcing(t, G.nodes[i], U_max)
            dR += gamma * U_i
            G.nodes[i]['risk'] = current_risks[i] + dR * dt
            risk_history[i].append(G.nodes[i]['risk'])
        total_risk = sum(G.nodes[i]['risk'] for i in G.nodes())
        total_risk_history.append(total_risk)
    return risk_history, total_risk_history

# ================================
# Module: Plotting & Animation
# ================================
def animate_simulation(G, risk_history, total_risk_history, params):
    pos = nx.spring_layout(G)
    fig, (ax_net, ax_risk) = plt.subplots(1, 2, figsize=(12,5))
    num_steps = len(total_risk_history)
    node_collection = None
    
    def update(step):
        nonlocal node_collection
        ax_net.clear()
        risks = [risk_history[node][step] for node in G.nodes()]
        node_collection = nx.draw_networkx_nodes(G, pos, node_color=risks, cmap=plt.cm.viridis, ax=ax_net)
        nx.draw_networkx_edges(G, pos, ax=ax_net)
        ax_net.set_title(f"Network Risk at t={step*params['dt']:.2f}")
        plt.colorbar(node_collection, ax=ax_net)
        
        ax_risk.clear()
        ax_risk.plot(np.arange(num_steps)*params['dt'], total_risk_history, 'r-')
        ax_risk.axhline(params['safety_threshold'], color='k', linestyle='--', label='Safety Threshold')
        ax_risk.set_title("Total Network Risk Over Time")
        ax_risk.set_xlabel("Time")
        ax_risk.set_ylabel("Total Risk")
        ax_risk.legend()
        
    ani = FuncAnimation(fig, update, frames=num_steps, interval=100)
    plt.show()

# ================================
# Input Handler: Choose Project Source (Text, File, URL)
# ================================
def get_project_source():
    input_type = widgets.RadioButtons(
        options=["Text", "File Upload", "URL"],
        description="Input Type:",
        disabled=False
    )
    text_area = widgets.Textarea(
        description="Project Source:",
        layout=widgets.Layout(width="600px", height="200px"),
        value="""# Define simulation parameters (one per line)
alpha = 0.5
beta = 0.1
gamma = 0.05
U_max = 1.0
safety_threshold = 10
num_nodes = 20
dt = 0.1
T = 20.0
"""
    )
    file_upload = widgets.FileUpload(accept=".txt,.py,.zip", multiple=False)
    url_text = widgets.Text(
        description="Project URL:",
        placeholder="Enter URL to a project script or parameter file"
    )
    container = widgets.VBox([text_area])
    
    def on_input_type_change(change):
        if change['new'] == "Text":
            container.children = [text_area]
        elif change['new'] == "File Upload":
            container.children = [file_upload]
        elif change['new'] == "URL":
            container.children = [url_text]
    
    input_type.observe(on_input_type_change, names="value")
    return input_type, container, text_area, file_upload, url_text

# ================================
# Function: Load project source from selected input
# ================================
def load_project_source(input_type, text_area, file_upload, url_text):
    if input_type.value == "Text":
        return text_area.value
    elif input_type.value == "File Upload":
        if file_upload.value:
            key = list(file_upload.value.keys())[0]
            file_info = file_upload.value[key]
            content = file_info['content']
            if file_info['metadata']['name'].endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    names = [n for n in z.namelist() if n.endswith(".txt")]
                    if names:
                        with z.open(names[0]) as f:
                            return f.read().decode("utf-8")
                    else:
                        return ""
            else:
                return content.decode("utf-8")
        else:
            return ""
    elif input_type.value == "URL":
        try:
            response = requests.get(url_text.value)
            response.raise_for_status()
            return response.text
        except Exception as e:
            return f"Error fetching URL: {e}"
    else:
        return ""

# ================================
# Module: Idea Expansion and Business Model Design
# ================================
def expand_idea(idea_text):
    prompt_expansion = (
        "Expand the following idea into a detailed business model. "
        "Describe all parts of the idea, propose a timeline, and design a safe and profitable system or method, "
        "including tools, campaigns, or gimmicks that could be used. "
        "Idea: " + idea_text
    )
    expansion = GeminiAI.generate_analysis(prompt_expansion, idea_text)
    return expansion

# ================================
# Idea Expansion GUI Module
# ================================
def run_idea_expansion():
    clear_output(wait=True)
    idea_input = widgets.Textarea(
        description="Idea:",
        layout=widgets.Layout(width="600px", height="150px"),
        value="Enter your idea here..."
    )
    expand_button = widgets.Button(description="Expand Idea")
    idea_output = widgets.Textarea(
        description="Expanded Idea:",
        layout=widgets.Layout(width="800px", height="300px")
    )
    
    display(idea_input, expand_button, idea_output)
    
    def on_expand_clicked(b):
        idea_text = idea_input.value
        if not idea_text.strip():
            idea_output.value = "Please enter a valid idea."
        else:
            idea_output.value = "Expanding idea, please wait..."
            expanded = expand_idea(idea_text)
            idea_output.value = expanded
    
    expand_button.on_click(on_expand_clicked)

# ================================
# Module: Optimization and Fixes
# ================================
def simulate_final_risk(params):
    """Utility to run a short simulation and return the final total risk."""
    G = build_network(params["num_nodes"])
    _, total_risk_history = simulate_risk_dynamics(G, params)
    return total_risk_history[-1]

def optimize_parameters(initial_params, target_risk, learning_rate=0.01, iterations=10):
    """
    A simple gradient-descent-like optimizer that adjusts parameters (alpha, beta, gamma)
    to reduce the difference between final total risk and a target risk.
    
    We use finite differences (vector math) to approximate the gradient.
    """
    x = np.array([initial_params["alpha"], initial_params["beta"], initial_params["gamma"]])
    param_keys = ["alpha", "beta", "gamma"]
    
    for it in range(iterations):
        current_params = initial_params.copy()
        for idx, key in enumerate(param_keys):
            current_params[key] = x[idx]
        final_risk = simulate_final_risk(current_params)
        loss = (final_risk - target_risk) ** 2
        
        grad = np.zeros_like(x)
        delta = 1e-4
        for i in range(len(x)):
            x_old = x[i]
            x[i] = x_old + delta
            temp_params = initial_params.copy()
            for idx, key in enumerate(param_keys):
                temp_params[key] = x[idx]
            final_risk_plus = simulate_final_risk(temp_params)
            loss_plus = (final_risk_plus - target_risk) ** 2
            grad[i] = (loss_plus - loss) / delta
            x[i] = x_old  # restore
        x = x - learning_rate * grad
        print(f"Iteration {it+1}: loss={loss:.4f}, parameters={x}, final_risk={final_risk:.4f}")
    
    optimized_params = initial_params.copy()
    for idx, key in enumerate(param_keys):
        optimized_params[key] = x[idx]
    return optimized_params, loss, grad

def optimize_simulation_and_explain(project_source):
    """
    Takes a project source (defining simulation parameters), runs the simulation,
    then uses our optimizer to adjust parameters (alpha, beta, gamma) so that the final total risk
    is as close as possible to the safety threshold. Finally, calls Gemini to explain the changes.
    """
    initial_params = parse_parameters(project_source)
    target_risk = initial_params["safety_threshold"]
    print("Initial simulation parameters:", initial_params)
    initial_risk = simulate_final_risk(initial_params)
    print(f"Initial final risk: {initial_risk:.4f} (target: {target_risk})")
    
    optimized_params, final_loss, grad = optimize_parameters(initial_params, target_risk, learning_rate=0.05, iterations=10)
    optimized_risk = simulate_final_risk(optimized_params)
    print("Optimized parameters:", optimized_params)
    print(f"Optimized final risk: {optimized_risk:.4f}")
    
    explanation_prompt = (
        "The initial simulation parameters were: " + str(initial_params) + ". "
        "After optimization, the parameters changed to: " + str(optimized_params) + ". "
        "The final risk changed from " + f"{initial_risk:.4f}" + " to " + f"{optimized_risk:.4f}" + ". "
        "Explain the reasoning and adjustments needed to achieve a safer system and why these parameter changes are beneficial."
    )
    explanation = GeminiAI.generate_analysis(explanation_prompt, optimized_params)
    return optimized_params, optimized_risk, explanation

# ================================
# Optimization GUI Module
# ================================
def run_optimization():
    clear_output(wait=True)
    opt_input_type, opt_source_container, opt_text_area, opt_file_upload, opt_url_text = get_project_source()
    target_risk_widget = widgets.FloatText(
        description="Target Risk:",
        value=10.0
    )
    optimize_button = widgets.Button(description="Optimize Parameters")
    optimization_output = widgets.Textarea(
        description="Optimization Output:",
        layout=widgets.Layout(width="800px", height="300px")
    )
    
    display(opt_input_type, opt_source_container, target_risk_widget, optimize_button, optimization_output)
    
    def on_optimize_clicked(b):
        project_source = load_project_source(opt_input_type, opt_text_area, opt_file_upload, opt_url_text)
        if not project_source:
            optimization_output.value = "No project source provided."
            return
        target_risk = target_risk_widget.value
        optimization_output.value = "Optimizing parameters, please wait...\n"
        optimized_params, optimized_risk, explanation = optimize_simulation_and_explain(project_source)
        out_str = (
            "Optimized Parameters:\n" + str(optimized_params) + "\n\n" +
            f"Optimized Final Risk: {optimized_risk:.4f} (Target: {target_risk})\n\n" +
            "Explanation from Gemini:\n" + explanation
        )
        optimization_output.value = out_str
    
    optimize_button.on_click(on_optimize_clicked)

# ================================
# Settings Tab
# ================================
def run_settings():
    clear_output(wait=True)
    api_key_input = widgets.Text(
        description="API Key:",
        value=global_settings.get("API_KEY", "YOUR_GEMINI_API_KEY")
    )
    engine_dropdown = widgets.Dropdown(
        options=["GPT-4", "Gemini", "grok2"],
        description="API Engine:",
        value=global_settings.get("API_ENGINE", "Gemini")
    )
    save_button = widgets.Button(description="Save Settings")
    settings_output = widgets.Textarea(
        description="Settings:",
        layout=widgets.Layout(width="800px", height="100px")
    )
    
    def on_save_clicked(b):
        global_settings["API_KEY"] = api_key_input.value
        global_settings["API_ENGINE"] = engine_dropdown.value
        settings_output.value = f"Settings saved:\nAPI Key: {global_settings['API_KEY']}\nAPI Engine: {global_settings['API_ENGINE']}"
    
    save_button.on_click(on_save_clicked)
    display(api_key_input, engine_dropdown, save_button, settings_output)

# ================================
# GitHub Repo Tab
# ================================
def run_github_repo():
    clear_output(wait=True)
    repo_url = widgets.Text(
        description="Repo URL:",
        placeholder="https://github.com/yourusername/yourrepo"
    )
    file_path = widgets.Text(
        description="File Path:",
        placeholder="path/to/yourfile.txt"
    )
    load_button = widgets.Button(description="Load File")
    repo_output = widgets.Textarea(
        description="Repo File Content:",
        layout=widgets.Layout(width="800px", height="300px")
    )
    
    def on_load_clicked(b):
        if not repo_url.value or not file_path.value:
            repo_output.value = "Please enter both a repo URL and a file path."
            return
        try:
            base = repo_url.value.rstrip('/')
            raw_url = base.replace("github.com", "raw.githubusercontent.com") + "/main/" + file_path.value
            response = requests.get(raw_url)
            response.raise_for_status()
            repo_output.value = response.text
        except Exception as e:
            repo_output.value = f"Error loading file: {e}"
    
    load_button.on_click(on_load_clicked)
    display(repo_url, file_path, load_button, repo_output)

# ================================
# Proposals Tab
# ================================
def run_proposals():
    clear_output(wait=True)
    current_version = widgets.Textarea(
        description="Current Version:",
        layout=widgets.Layout(width="600px", height="150px"),
        value="Enter current system parameters or description here..."
    )
    proposal_version = widgets.Textarea(
        description="Proposed Version:",
        layout=widgets.Layout(width="600px", height="150px"),
        value="Enter proposed changes or version here..."
    )
    compare_button = widgets.Button(description="Compare Versions")
    proposals_output = widgets.Textarea(
        description="Comparison Output:",
        layout=widgets.Layout(width="800px", height="300px")
    )
    
    def on_compare_clicked(b):
        curr_text = current_version.value.splitlines()
        prop_text = proposal_version.value.splitlines()
        diff = difflib.unified_diff(curr_text, prop_text, fromfile='Current Version', tofile='Proposed Version', lineterm="")
        diff_text = "\n".join(list(diff))
        diff_lines = len(diff_text.splitlines())
        proposals_output.value = f"Diff between versions ({diff_lines} lines changed):\n\n" + diff_text
        
        explanation = GeminiAI.generate_analysis("Explain the impact of the proposed changes compared to the current version", diff_text)
        proposals_output.value += "\n\nImpact Explanation:\n" + explanation
    
    compare_button.on_click(on_compare_clicked)
    display(current_version, proposal_version, compare_button, proposals_output)

# ================================
# Main GUI Application Flow (Tabs)
# ================================
def run_main_application():
    tab = widgets.Tab()
    
    # --- Simulation Tab ---
    sim_network_name = widgets.Text(value="My Network", description="Network Name:")
    sim_input_type, sim_source_container, sim_text_area, sim_file_upload, sim_url_text = get_project_source()
    sim_process_button = widgets.Button(description="Run Simulation")
    sim_output_area = widgets.Output()
    sim_tab = widgets.VBox([
        sim_network_name,
        sim_input_type,
        sim_source_container,
        sim_process_button,
        sim_output_area
    ])
    
    def on_simulation_clicked(b):
        with sim_output_area:
            clear_output()
            project_source = load_project_source(sim_input_type, sim_text_area, sim_file_upload, sim_url_text)
            if not project_source:
                print("No project source provided.")
                return
            analysis = GeminiAI.generate_analysis("Analyzing simulation source", project_source)
            print(analysis)
            params = parse_parameters(project_source)
            print("Using simulation parameters:", params)
            G = build_network(params["num_nodes"])
            risk_history, total_risk_history = simulate_risk_dynamics(G, params)
            final_total_risk = total_risk_history[-1]
            print(f"Final Total Risk: {final_total_risk:.3f} (Safety threshold = {params['safety_threshold']})")
            if final_total_risk > params["safety_threshold"]:
                print("WARNING: Safety threshold exceeded!")
            else:
                print("Safety conditions maintained.")
            animate_simulation(G, risk_history, total_risk_history, params)
    
    sim_process_button.on_click(on_simulation_clicked)
    
    # --- Idea Expansion Tab ---
    idea_tab_button = widgets.Button(description="Run Idea Expansion")
    idea_tab = widgets.VBox([idea_tab_button])
    
    def on_idea_tab_clicked(b):
        run_idea_expansion()
    
    idea_tab_button.on_click(on_idea_tab_clicked)
    
    # --- Optimization Tab ---
    opt_tab_button = widgets.Button(description="Run Parameter Optimization")
    opt_tab = widgets.VBox([opt_tab_button])
    
    def on_opt_tab_clicked(b):
        run_optimization()
    
    opt_tab_button.on_click(on_opt_tab_clicked)
    
    # --- Settings Tab ---
    settings_tab_button = widgets.Button(description="Settings")
    settings_tab = widgets.VBox([settings_tab_button])
    
    def on_settings_tab_clicked(b):
        run_settings()
    
    settings_tab_button.on_click(on_settings_tab_clicked)
    
    # --- GitHub Repo Tab ---
    github_tab_button = widgets.Button(description="GitHub Repo")
    github_tab = widgets.VBox([github_tab_button])
    
    def on_github_tab_clicked(b):
        run_github_repo()
    
    github_tab_button.on_click(on_github_tab_clicked)
    
    # --- Proposals Tab ---
    proposals_tab_button = widgets.Button(description="Proposals")
    proposals_tab = widgets.VBox([proposals_tab_button])
    
    def on_proposals_tab_clicked(b):
        run_proposals()
    
    proposals_tab_button.on_click(on_proposals_tab_clicked)
    
    tab.children = [sim_tab, idea_tab, opt_tab, settings_tab, github_tab, proposals_tab]
    tab.set_title(0, "Simulation")
    tab.set_title(1, "Idea Expansion")
    tab.set_title(2, "Optimization")
    tab.set_title(3, "Settings")
    tab.set_title(4, "GitHub Repo")
    tab.set_title(5, "Proposals")
    
    display(tab)

# ================================
# Main Application Entry Point
# ================================
def main():
    run_main_application()

# Run the application if in a Jupyter environment.
if __name__ == "__main__":
    main()
