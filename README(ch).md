# DSL领域建模统一框架

本项目定义了一个统一、模块化的框架，用于构建和执行面向特定领域的 DSL（领域特定语言）。目前已支持统计领域，后续可扩展到医学、音乐、基因编辑等多个方向。

每种 DSL 都遵循以下统一生命周期：  
**DSL 代码 - 解析（Parsing） - 抽象语法树（AST） - 校验（Validation） - 编译/执行**

## 🌐 项目结构总览

```
project_root/
 │
 ├── core/                     # 所有 DSL 共享的基类定义
 │   ├── dataclass.py          # 定义 DSLProgram 的抽象基类
 │   ├── parser.py             # 基础解析器与 AST 构造器（基于 Lark）
 │   └── compiler.py           # 编译器的抽象基类
 │
 ├── statdsl/                  # 一个具体的统计学 DSL 实现
 │   ├── grammar.md            # 用于人类阅读的语法说明文档
 │   ├── statdsl.ebnf.txt      # EBNF 语法规范，供 Lark 使用
 │   ├── statprogram.py        # DSLProgram 的统计子类
 │   ├── statparser.py         # 将语法树转换为 AST 的 transformer
 │   └── statcompiler.py       # 编译器，将 DSL 编译为 Stan 代码
 │
 └── other DSL ...             # 其他 DSL 实现模块
```

---

## 📦 `core/` 模块说明

`core` 文件夹下定义的是所有 DSL 通用的基础抽象类。

### `dataclass.py`

- 定义抽象类 `DSLProgram`。
- 所有具体领域的 DSL 程序都需要继承这个类，并实现 `validate()` 方法用于程序结构合法性检查。

### `parser.py`

- 提供两个基础类：
  - `DSLParser`：加载 EBNF 语法，并将 DSL 源码解析为语法树。
  - `DSLTransformer`：将语法树转换为 AST 或对应的数据结构。
- 每个 DSL 都应继承 `DSLTransformer`，定义具体的节点处理逻辑。

### `compiler.py`

- 提供抽象类 `DSLCompiler`：
  - 负责组织完整的编译流程：解析 - 转换 - 验证 - 编译。
  - 子类需要实现 `_compile(program)` 方法，将程序转为目标代码（例如 Python/SQL/Stan）。

---

## 🧪 `statdsl/`：统计建模 DSL 示例

本模块实现了一个用于编写统计模型的 DSL，可以将 DSL 编译为 Stan 代码并执行。

- `grammar.md`：用于人类阅读的 DSL 语法文档。
- `statdsl.ebnf.txt`：定义 DSL 格式的 EBNF 文法文件，供 Lark 使用。
- `statprogram.py`：定义统计 DSL 的 AST 结构，实现自 `DSLProgram`。
- `statparser.py`：解析器的转换逻辑，将语法树转为 `StatProgram`。
- `statcompiler.py`：编译器，将 AST 编译为可执行的 Stan 代码。

---

## 🧑‍💻 如何添加你自己的 DSL 模块

如果你要添加一个新的 DSL（如医学、音乐、基因编辑等），请按照以下步骤操作：

### 1. 创建一个新的模块文件夹

例如：`meddsl/`, `musicdsl/`, `genedsl/` 等。

### 2. 定义 DSL 的语法规则

- 使用 EBNF 格式写一个语法定义文件（例如 `yourdsl.ebnf.txt`）。
- 可选：提供一份用于人类阅读的语法说明文档（例如 `grammar.md`）。

### 3. 实现三大核心组件

在你的 DSL 模块文件夹中：

- `yourprogram.py`：
  - 实现自 `DSLProgram`，定义你自己的语义结构。
  - 实现 `validate()` 方法，确保程序合法性。

- `yourparser.py`：
  - 实现自 `DSLTransformer`，定义每个语法节点如何转为 AST。

- `yourcompiler.py`：
  - 实现自 `DSLCompiler`，定义如何将 AST 编译为目标语言（如 SQL、Python、LaTeX、可视化等）。

### 4. 测试集成流程

- 创建对应的 parser、transformer、compiler 实例。
- 输入 DSL 源码，执行完整流程，检查是否能成功转换和执行。

---

## 最小示例入口

```python
from core import DSLParser
from statdsl import StatModelTrans, StatModelCompiler

grammar_file = "statdsl/statdsl_v1_0.ebnf.txt"
start_symbol = "stat_model_spec"
parser = DSLParser(grammar_file=grammar_file, start_symbol=start_symbol)
trans = StatModelTrans()
compiler = StatModelCompiler(parser, trans)

dsl_code = """
model linear_regression {
  data {
    real x;
    real y;
  }
  parameters {
    real beta;
  }
  model {
    y ~ normal(beta * x, 1);
  }
}
"""

compiled_output = compiler.compile(dsl_code)
print(compiled_output)
```

------

## 💡 说明

- 每种 DSL 都是独立模块，但共享统一的抽象接口与流程。
- 每个 DSL 的 `validate()` 方法用于进行结构合法性检查，是代码生成前的重要步骤。
- 框架可扩展，支持插件式集成解释器、优化器、模拟器等工具。

