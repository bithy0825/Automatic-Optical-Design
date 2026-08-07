"""全库唯一词表：配置键与参数名的规范形式及别名。

每个 :class:`~core.noun.Noun` 声明一个术语的规范形式（canonical）与全部
别名；配置解析、属性注册、变异参数查找一律经由本词表，禁止在业务代码中
散落硬编码字符串。
"""

from typing import Final

from core.noun import Noun

# ── 种群 / 遗传算法 ──
POPULATION: Final = Noun.for_key("Population", "population")
TOPK: Final = Noun.for_key("Top_K", "topk")
GENERATION: Final = Noun.for_key("Generation", "generation")
GA: Final = Noun.for_key("Genetic_Algorithm", "GA", "ga")
MUTATE: Final = Noun.for_key("Mutate", "mutate")
BOUNDS: Final = Noun.for_key("Bounds", "bounds")

# ── 优化器 ──
OPTIMIZER: Final = Noun.for_key("Optimizer", "optimizer")
STEP: Final = Noun.for_key("Step", "step")
LR: Final = Noun.for_key("Learning_Rate", "lr")
SCHEDULER: Final = Noun.for_key("Scheduler", "scheduler")
GRAD_NORM: Final = Noun.for_key("Gradient_Norm", "grad_norm")
DEFAULT: Final = Noun.for_key("Default", "default")

# ── 训练运行控制（[train] 节） ──
SEED: Final = Noun.for_key("Seed", "seed")
DEVICE: Final = Noun.for_key("Device", "device")
OUTPUT: Final = Noun.for_key("Output", "output")
RESUME: Final = Noun.for_key("Resume", "resume")
SAVE_EVERY: Final = Noun.for_key("Save_Every", "save_every")
HISTORY: Final = Noun.for_key("History", "history")

# ── 目标与采样 ──
TARGET: Final = Noun.for_key("Target", "target")
FOV: Final = Noun.for_key("Field_of_View", "FOV", "fov")
F_NUMBER: Final = Noun.for_key("F_Number", "F", "F#")
EFFL: Final = Noun.for_key("Effective_Focal_Length", "EFFL", "effl")
WAVELENGTH: Final = Noun.for_key("Wavelength", "wavelength", "λ")
WAVEL: Final = Noun.for_key("Wavel", "wavel")  # 光源的波长采样配置块键
EPD: Final = Noun.for_key("Entrance_Pupil_Diameter", "EPD", "epd")
PUPIL: Final = Noun.for_key("Pupil", "pupil")
FIELD: Final = Noun.for_key("Field", "field")
REGION: Final = Noun.for_key("Region", "region")
COUNT: Final = Noun.for_key("Count", "count")

# ── 系统结构 ──
ID: Final = Noun.for_key("ID", "id")
COMPONENT: Final = Noun.for_key("Component", "component")
TYPE: Final = Noun.for_key("Type", "type")
METHOD: Final = Noun.for_key("Method", "method")
INITIAL: Final = Noun.for_key("Initial", "initial")
RAW: Final = Noun.for_key("Raw", "raw")
VALUE: Final = Noun.for_key("Value", "value")
UNIFORM: Final = Noun.for_key("Uniform", "uniform")
NORMAL: Final = Noun.for_key("Normal", "normal")
RANDOM: Final = Noun.for_key("Random", "random")
MEAN: Final = Noun.for_key("Mean", "mean")
STD: Final = Noun.for_key("Standard_Deviation", "std")
LOW: Final = Noun.for_key("Low", "low")
HIGH: Final = Noun.for_key("High", "high")
TRAIN: Final = Noun.for_key("Train", "train")
INDEX: Final = Noun.for_key("Index", "index")
INDICES: Final = Noun.for_key("Indices", "indices")
# ── 损失与裁决死因（LossWeights 键） ──
LOSS: Final = Noun.for_key("Loss", "loss")
BLUR: Final = Noun.for_key("Blur", "blur")
DISTORTION: Final = Noun.for_key("Distortion", "distortion")
SAG_DOMAIN: Final = Noun.for_key("Sag_Domain", "sag_domain")
SOLVER_NEGATIVE: Final = Noun.for_key("Solver_Negative", "solver_negative")
SOLVER_CONVERGENCE: Final = Noun.for_key("Solver_Convergence", "solver_convergence")
APERTURE_CLIP: Final = Noun.for_key("Aperture_Clip", "aperture_clip")
TIR: Final = Noun.for_key("TIR", "tir")

# ── 元件种类（Component.kind） ──
SOURCE: Final = Noun.for_key("Source", "source")
GAP: Final = Noun.for_key("Gap", "gap")
REFRACTOR: Final = Noun.for_key("Refractor", "refractor")
STOP: Final = Noun.for_key("Stop", "stop")
SENSOR: Final = Noun.for_key("Sensor", "sensor")
SEQUENTIAL: Final = Noun.for_key("Sequential", "sequential")

# ── 面形种类（Shape.kind）与几何参数 ──
SHAPE: Final = Noun.for_key("Shape", "shape")
SPHERE: Final = Noun.for_key("Sphere", "sphere")
CONIC: Final = Noun.for_key("Conic", "conic")
ASPHERE: Final = Noun.for_key("Asphere", "asphere")
DISK: Final = Noun.for_key("Disk", "disk")
DIAMETER: Final = Noun.for_key("Diameter", "diameter")
RADIUS: Final = Noun.for_key("Radius", "radius")
CURVATURE: Final = Noun.for_key("Curvature", "curvature")
KAPPA: Final = Noun.for_key("Kappa", "kappa")
ALPHA: Final = Noun.for_key("Alpha", "alpha")
MASK: Final = Noun.for_key("Mask", "mask")
THICKNESS: Final = Noun.for_key("Thickness", "thickness")
SOLVER: Final = Noun.for_key("Solver", "solver")

# ── 材料 ──
MATERIAL: Final = Noun.for_key("Material", "material")
DATABASE: Final = Noun.for_key("Database", "database", "db")
CONSTANT: Final = Noun.for_key("Constant", "constant")
SELLMEIER: Final = Noun.for_key("Sellmeier", "sellmeier")
