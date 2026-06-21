import torch  # Hlavní knihovna pro tenzory a neuronové sítě
import torch.nn as nn  # Modul pro definici vrstev sítě
import torch.optim as optim  # Modul pro optimalizační algoritmy (Adam)
import pandas as pd  # Knihovna pro manipulaci s CSV daty
import json  # Pro načtení konfiguračního souboru s alfou

# Automaticky vybere GPU (CUDA), pokud je dostupné, jinak použije CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    with open("constants.json", "r") as f:
        constants = json.load(f)  # Načte alfa konstantu
        alpha = constants["alpha"]

    # Načtení CSV souborů do Pandas DataFrame
    train = pd.read_csv("train_measurements.csv")
    boundary = pd.read_csv("boundary_partial.csv")
    initial = pd.read_csv("initial_sparse.csv")
    test = pd.read_csv("test_points.csv")

    # Pomocná funkce: převede sloupce DataFrame na tensor tvaru [N, k]
    # a odešle ho na zařízení (GPU/CPU). Tvar [N, 1] je nutný pro
    # správné násobení matic v síti.
    def to_tensor(df, cols):
        return torch.tensor(df[cols].values, dtype=torch.float32).to(device)

    return {
        "train": (to_tensor(train, ["x", "t"]), to_tensor(train, ["u"])),
        "boundary": (to_tensor(boundary, ["x", "t"]), to_tensor(boundary, ["u"])),
        "initial": (to_tensor(initial, ["x", "t"]), to_tensor(initial, ["u"])),
        "test": to_tensor(test, ["x", "t"]),
        "test_ids": test["id"].values,
        "alpha": alpha,
    }


class HeatNet(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        # Definice sítě: 3 skryté vrstvy, Tanh jako aktivace
        # Tanh je hladká funkce (derivovatelná), což je pro PINN klíčové.
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),  # Výstup: teplota u
        )

    def forward(self, x, t):
        # Spojí x a t do jednoho vektoru [N, 2] pro vstup do sítě
        return self.net(torch.cat([x, t], dim=1))


def loss_data(model, x, t, u_true):
    # Klasické MSE mezi predikcí sítě a známými daty
    return torch.mean((model(x, t) - u_true) ** 2)


def loss_physics(model, x_col, t_col, alpha):
    # Aktivace sledování gradientů pro kolokační body (nutné pro derivaci)
    x_col.requires_grad_(True)
    t_col.requires_grad_(True)

    u = model(x_col, t_col)  # Predikce teploty

    # Autograd: Vypočítá derivace pomocí zpětného průchodu grafem
    # create_graph=True umožňuje derivovat i samotný výsledek derivace
    u_t = torch.autograd.grad(
        u, t_col, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_x = torch.autograd.grad(
        u, x_col, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x_col, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]

    # PDE rovnice: du/dt = alpha * d^2u/dx^2 => residual = du/dt - alpha * u_xx
    residual = u_t - alpha * u_xx
    # Cílem je, aby residual byl 0 (rovnice byla splněna)
    return torch.mean(residual**2)


def train(model, data, epochs=5001):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Váhy jednotlivých složek ztráty (vážený součet)
    w_data, w_bc, w_ic, w_phys = 1.0, 1.0, 1.0, 0.1

    x_train, u_train = data["train"]
    x_bc, u_bc = data["boundary"]
    x_ic, u_ic = data["initial"]
    alpha = data["alpha"]

    for epoch in range(epochs):
        optimizer.zero_grad()  # Vynulování gradientů z minulého kroku

        # Výpočet ztrát z dat a podmínek (boundary/initial)
        l_d = loss_data(model, x_train[:, 0:1], x_train[:, 1:2], u_train)
        l_bc = loss_data(model, x_bc[:, 0:1], x_bc[:, 1:2], u_bc)
        l_ic = loss_data(model, x_ic[:, 0:1], x_ic[:, 1:2], u_ic)

        # Generování náhodných bodů v doméně pro "fyzikální dozor"
        x_col = torch.rand(2000, 1, device=device)
        t_col = torch.rand(2000, 1, device=device)
        l_phys = loss_physics(model, x_col, t_col, alpha)

        # Celková loss (vážený součet)
        loss = w_data * l_d + w_bc * l_bc + w_ic * l_ic + w_phys * l_phys

        loss.backward()  # Výpočet gradientů (backpropagation)
        optimizer.step()  # Aktualizace vah sítě

        if epoch % 500 == 0:
            print(
                f"Epoch {epoch:5d} | loss={loss.item():.6f} "
                f"| data={l_d.item():.6f} bc={l_bc.item():.6f} "
                f"ic={l_ic.item():.6f} phys={l_phys.item():.6f}"
            )


def generate_submission(model, data):
    model.eval()  # Přepnutí do módu evaluace (vypne např. dropout)
    with torch.no_grad():  # Vypne sledování gradientů (šetří paměť)
        test_inputs = data["test"]
        preds = model(test_inputs[:, 0:1], test_inputs[:, 1:2]).cpu().numpy()

    # Formátování do DataFrame a uložení do CSV
    df = pd.DataFrame({"id": data["test_ids"], "u": preds.flatten()})
    df.to_csv("submission.csv", index=False)
    print("Uloženo do submission.csv")


def main():
    data = load_data()
    model = HeatNet().to(device)
    train(model, data)
    generate_submission(model, data)


if __name__ == "__main__":
    main()
