#!/usr/bin/env python3

import os
import sys
import json
import shlex
import subprocess
from pathlib import Path
import urllib.request
import urllib.error


# ============================================================
# CONFIGURAÇÃO
# ============================================================

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nvidia/nemotron-3.5-lightning:free"

PROJECT = Path.home() / "cartola-predicao-v2"

MAX_OUTPUT = 12000
MAX_TURNS = 30

# Limite de chamadas nesta execução.
MAX_API_CALLS = 30

API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not API_KEY:
    print("ERRO: OPENROUTER_API_KEY não está definida.")
    print()
    print('export OPENROUTER_API_KEY="SUA_CHAVE"')
    sys.exit(1)

if not PROJECT.is_dir():
    print(f"ERRO: projeto não encontrado: {PROJECT}")
    sys.exit(1)


# ============================================================
# COMANDOS
# ============================================================

# Comandos considerados somente leitura.
READ_ONLY = {
    "pwd",
    "ls",
    "find",
    "du",
    "df",
    "file",
    "stat",
    "head",
    "tail",
    "cat",
    "grep",
    "egrep",
    "fgrep",
    "sort",
    "uniq",
    "wc",
    "cut",
    "awk",
    "sed",
    "diff",
    "sha256sum",
    "which",
    "command",
    "realpath",
    "readlink",
    "uname",
    "date",
    "tree",
}


# Programas sempre bloqueados.
BLOCKED_PROGRAMS = {
    "sudo",
    "su",
    "doas",
    "rm",
    "rmdir",
    "shred",
    "mkfs",
    "dd",
    "mount",
    "umount",
    "reboot",
    "shutdown",
    "poweroff",
    "kill",
    "pkill",
    "killall",
    "passwd",
    "chown",
    "chgrp",
}


# Operações Git explicitamente proibidas.
BLOCKED_GIT = {
    ("reset", "--hard"),
    ("clean",),
    ("push",),
    ("rebase",),
}


# Strings perigosas relacionadas a execução remota.
BLOCKED_PATTERNS = [
    "curl | sh",
    "curl|sh",
    "curl | bash",
    "curl|bash",
    "curl | zsh",
    "curl|zsh",
    "wget | sh",
    "wget|sh",
    "wget | bash",
    "wget|bash",
    "wget | zsh",
    "wget|zsh",
    "| sh",
    "|sh",
    "| bash",
    "|bash",
    "| zsh",
    "|zsh",
    "/dev/tcp/",
]


# ============================================================
# UTILITÁRIOS DE SEGURANÇA
# ============================================================

def inside_project(path: Path) -> bool:
    """
    Verifica se um caminho está dentro do projeto.
    """
    try:
        path.resolve().relative_to(PROJECT.resolve())
        return True
    except ValueError:
        return False


def has_dangerous_redirection(command: str) -> bool:
    """
    Bloqueia redirecionamentos de escrita reais.

    Permite:
        2>/dev/null
        2>/dev/null

    Bloqueia:
        > arquivo
        >> arquivo
        1> arquivo
        2> arquivo
        &> arquivo
    """

    import re

    # stderr descartado é seguro.
    normalized = command.replace(" ", "")

    safe_stderr = {
        "2>/dev/null",
        "2>/dev/null|",
    }

    # Removemos temporariamente ocorrências seguras.
    cleaned = normalized.replace("2>/dev/null", "")

    # Qualquer > restante é considerado potencial escrita.
    if ">" in cleaned:
        return True

    return False


def has_unsafe_pipeline(command: str) -> bool:
    """
    Permite pipelines de leitura:

        find ... | head
        grep ... | head
        ls ... | sort

    Bloqueia pipelines que terminam executando código.
    """

    lowered = command.lower()

    for pattern in BLOCKED_PATTERNS:
        if pattern.lower() in lowered:
            return True

    return False


def classify_command(command: str):
    """
    Retorna:

        ("allow", motivo)
        ("ask", motivo)
        ("block", motivo)
    """

    text = command.strip()

    if not text:
        return ("block", "comando vazio")

    lowered = text.lower()

    # --------------------------------------------------------
    # Operações perigosas
    # --------------------------------------------------------

    if has_unsafe_pipeline(text):
        return (
            "block",
            "pipeline potencialmente executa código remoto ou shell"
        )

    if has_dangerous_redirection(text):
        return (
            "ask",
            "redirecionamento pode modificar arquivos"
        )

    # Não permitir execução de múltiplos comandos arbitrários.
    #
    # Pipeline "|" é permitido e tratado separadamente.
    forbidden_shell_operators = [
        "&&",
        "||",
        ";",
        "`",
        "$(",
    ]

    for op in forbidden_shell_operators:
        if op in text:
            return (
                "block",
                f"operador de shell não permitido: {op}"
            )

    # --------------------------------------------------------
    # Parse
    # --------------------------------------------------------

    try:
        args = shlex.split(text)
    except ValueError as e:
        return ("block", f"comando inválido: {e}")

    if not args:
        return ("block", "comando vazio")

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    if "|" in text:
        parts = text.split("|")

        if len(parts) > 4:
            return ("block", "pipeline excessivamente complexo")

        # Cada estágio precisa ser um comando conhecido.
        for part in parts:
            part = part.strip()

            if not part:
                return ("block", "pipeline inválido")

            try:
                stage_args = shlex.split(part)
            except ValueError:
                return ("block", "pipeline inválido")

            if not stage_args:
                return ("block", "pipeline inválido")

            program = Path(stage_args[0]).name

            if program in BLOCKED_PROGRAMS:
                return (
                    "block",
                    f"programa bloqueado dentro do pipeline: {program}"
                )

            if program not in READ_ONLY:
                return (
                    "ask",
                    f"pipeline contém comando não classificado: {program}"
                )

        return (
            "allow",
            "pipeline composto somente por comandos de leitura"
        )

    # --------------------------------------------------------
    # Programa
    # --------------------------------------------------------

    program = Path(args[0]).name

    if program in BLOCKED_PROGRAMS:
        return (
            "block",
            f"programa bloqueado: {program}"
        )

    # --------------------------------------------------------
    # Git
    # --------------------------------------------------------

    if program == "git":

        if len(args) < 2:
            return ("allow", "git sem subcomando")

        sub = args[1]

        # Operações destrutivas.
        if sub == "reset" and "--hard" in args:
            return (
                "block",
                "git reset --hard é proibido"
            )

        if sub == "clean":
            return (
                "block",
                "git clean é proibido"
            )

        if sub == "push":
            return (
                "block",
                "git push é proibido pelo agente"
            )

        if sub == "rebase":
            return (
                "block",
                "git rebase é proibido pelo agente"
            )

        safe_git = {
            "status",
            "log",
            "diff",
            "show",
            "branch",
            "tag",
            "remote",
            "rev-parse",
            "ls-files",
            "ls-tree",
        }

        if sub in safe_git:
            return (
                "allow",
                "consulta Git somente leitura"
            )

        return (
            "ask",
            f"operação Git potencialmente modificadora: {sub}"
        )

    # --------------------------------------------------------
    # Python
    # --------------------------------------------------------

    if program in {"python", "python3"}:

        if len(args) >= 2 and args[1] in {"-c", "-m"}:
            return (
                "ask",
                "Python inline/módulo pode modificar o projeto"
            )

        if len(args) >= 2:

            candidate = Path(args[1])

            if not candidate.is_absolute():
                candidate = PROJECT / candidate

            if not inside_project(candidate):
                return (
                    "block",
                    "script Python fora do projeto"
                )

            if candidate.exists():
                return (
                    "allow",
                    "execução de script existente dentro do projeto"
                )

        return (
            "ask",
            "Python sem script explícito"
        )

    # --------------------------------------------------------
    # Testes
    # --------------------------------------------------------

    if program in {
        "pytest",
        "pytest-asyncio",
    }:
        return (
            "allow",
            "execução de testes"
        )

    # --------------------------------------------------------
    # Leitura
    # --------------------------------------------------------

    if program in READ_ONLY:
        return (
            "allow",
            "comando considerado somente leitura"
        )

    # --------------------------------------------------------
    # Desconhecido
    # --------------------------------------------------------

    return (
        "ask",
        f"comando não classificado automaticamente: {program}"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

def execute_command(command: str):

    classification, reason = classify_command(command)

    print()
    print("=" * 70)
    print("COMANDO PROPOSTO")
    print("=" * 70)
    print(command)
    print()
    print("Classificação:", classification)
    print("Motivo:", reason)

    # --------------------------------------------------------
    # Bloqueado
    # --------------------------------------------------------

    if classification == "block":

        return {
            "status": "blocked",
            "command": command,
            "reason": reason,
        }

    # --------------------------------------------------------
    # Confirmação
    # --------------------------------------------------------

    if classification == "ask":

        print()

        answer = input(
            "Permitir este comando? [y/N]: "
        ).strip().lower()

        if answer not in {
            "y",
            "yes",
            "s",
            "sim",
        }:

            return {
                "status": "denied",
                "command": command,
                "reason": "usuário não autorizou",
            }

    # --------------------------------------------------------
    # Execução
    # --------------------------------------------------------

    try:

        # Pipeline seguro precisa ser executado pelo shell.
        # Mas só chegamos aqui se todos os estágios foram
        # classificados como comandos de leitura.
        if "|" in command:

            result = subprocess.run(
                command,
                cwd=PROJECT,
                shell=True,
                executable="/system/bin/sh",
                capture_output=True,
                text=True,
                timeout=300,
                env=os.environ.copy(),
            )

        else:

            args = shlex.split(command)

            result = subprocess.run(
                args,
                cwd=PROJECT,
                capture_output=True,
                text=True,
                timeout=300,
                env=os.environ.copy(),
            )

        stdout = result.stdout[-MAX_OUTPUT:]
        stderr = result.stderr[-MAX_OUTPUT:]

        return {
            "status": "executed",
            "command": command,
            "returncode": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }

    except subprocess.TimeoutExpired:

        return {
            "status": "timeout",
            "command": command,
            "reason": "limite de 300 segundos atingido",
        }

    except Exception as e:

        return {
            "status": "error",
            "command": command,
            "reason": repr(e),
        }


# ============================================================
# PROMPT DO NEMOTRON
# ============================================================

SYSTEM_PROMPT = r"""
Você é o agente executor do projeto Cartola AI Insights.

DIRETÓRIO DE TRABALHO:

~/cartola-predicao-v2

Você trabalha em um projeto de machine learning para Cartola FC.

============================================================
REGRAS ABSOLUTAS
============================================================

1. Trabalhe SOMENTE dentro do diretório do projeto.

2. Não use sudo, su ou doas.

3. Não remova arquivos.

4. Não execute git reset --hard.

5. Não execute git clean.

6. Não execute git push.

7. Não execute git rebase.

8. Não altere o CatBoost v2 oficial sem autorização explícita
   do usuário.

9. Não sobrescreva datasets ou artefatos oficiais sem autorização
   explícita.

10. Não invente resultados.

11. Antes de modificar qualquer coisa, primeiro inspecione o estado
    atual.

12. Prefira scripts existentes.

13. Use o menor número razoável de comandos.

14. Evite ficar executando vários ls consecutivos quando um find,
    grep ou uma inspeção mais abrangente resolver a tarefa.

15. Se a tarefa puder ser concluída somente por leitura/teste,
    não tente modificar arquivos.

16. Se uma alteração for realmente necessária, proponha o comando.
    O executor pedirá confirmação antes da alteração.

17. Nunca tente escapar do diretório do projeto.

18. Nunca exponha a OPENROUTER_API_KEY.

============================================================
SOBRE O EXP-12
============================================================

IMPORTANTE:

O projeto possui diferentes materiais relacionados a "exp12".

Não confunda:

reports/exp12/api_vs_csv_2026/

com o EXP-12 metodológico de contexto de time/adversário.

O EXP-12 metodológico trabalha com:

- histórico de partidas do time;
- histórico do adversário;
- gols marcados;
- gols sofridos;
- finalizações;
- finalizações certas;
- janelas temporais de 5 e 10 jogos;
- cobertura das janelas;
- prevenção de leakage temporal.

O auditor correto deve verificar que essas features usam SOMENTE
informações disponíveis antes da rodada-alvo.

============================================================
RESPOSTA
============================================================

Responda SEMPRE com JSON válido e nada além do JSON.

Para executar comando:

{
  "type": "command",
  "command": "comando",
  "reason": "por que este comando é necessário"
}

Quando concluir:

{
  "type": "done",
  "summary": "resumo objetivo",
  "evidence": [
    "evidência 1",
    "evidência 2"
  ]
}

Se precisar perguntar algo:

{
  "type": "question",
  "question": "pergunta"
}

Não use markdown.
"""


# ============================================================
# OPENROUTER
# ============================================================

def call_model(messages):

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1200,
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://openrouter.ai/",
            "X-Title": "Cartola AI Nemotron Agent",
        },
        method="POST",
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=120
        ) as response:

            raw = response.read().decode("utf-8")

        obj = json.loads(raw)

        return obj["choices"][0]["message"]["content"]

    except urllib.error.HTTPError as e:

        body = e.read().decode(
            "utf-8",
            errors="replace"
        )

        raise RuntimeError(
            f"OpenRouter HTTP {e.code}: {body}"
        )

    except Exception as e:

        raise RuntimeError(
            f"Erro chamando OpenRouter: {e}"
        )


# ============================================================
# PARSER
# ============================================================

def parse_model_response(text):

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if "```" in text:

        parts = text.split("```")

        for part in parts:

            cleaned = part.strip()

            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    raise ValueError(
        "Nemotron não retornou JSON válido.\n"
        f"Resposta recebida:\n{text}"
    )


# ============================================================
# AGENTE
# ============================================================

def main():

    task = " ".join(sys.argv[1:]).strip()

    if not task:

        print()
        print("Uso:")
        print()
        print(
            'python3 nemotron_agent.py "sua tarefa"'
        )
        print()

        sys.exit(1)

    print("=" * 70)
    print("NEMOTRON CARTOLA AGENT")
    print("=" * 70)

    print(
        "Modelo:",
        MODEL
    )

    print(
        "Projeto:",
        PROJECT
    )

    print()
    print("TAREFA:")
    print(task)

    print()
    print("Proteções ativas:")
    print("- comandos perigosos bloqueados")
    print("- operações desconhecidas pedem confirmação")
    print("- pipelines somente-leitura permitidos")
    print("- sem sudo")
    print("- sem remoção de arquivos")
    print("- sem git reset/clean/push/rebase")
    print("- cwd fixo no projeto")
    print(f"- máximo de {MAX_API_CALLS} chamadas nesta execução")
    print()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "TAREFA DO USUÁRIO:\n"
                + task
                + "\n\n"
                "Comece inspecionando o estado atual do projeto. "
                "Use o menor número razoável de comandos. "
                "Não modifique nada sem necessidade."
            ),
        },
    ]

    api_calls = 0

    for turn in range(
        1,
        MAX_TURNS + 1
    ):

        if api_calls >= MAX_API_CALLS:

            print()
            print("=" * 70)
            print("LIMITE DE CHAMADAS DA EXECUÇÃO ATINGIDO")
            print("=" * 70)
            print(
                f"Chamadas usadas: {api_calls}/{MAX_API_CALLS}"
            )

            return

        print()
        print(
            f"--- turno {turn}/{MAX_TURNS} "
            f"| API {api_calls + 1}/{MAX_API_CALLS} ---"
        )

        try:

            api_calls += 1

            response = call_model(messages)

        except Exception as e:

            print("ERRO:", e)

            return

        print()
        print("NEMOTRON:")
        print(response)

        try:

            action = parse_model_response(
                response
            )

        except Exception as e:

            print()
            print("ERRO:", e)
            print("Encerrando por segurança.")

            return

        action_type = action.get("type")

        # ----------------------------------------------------
        # COMANDO
        # ----------------------------------------------------

        if action_type == "command":

            command = action.get(
                "command",
                ""
            ).strip()

            if not command:

                print(
                    "ERRO: comando vazio."
                )

                return

            result = execute_command(
                command
            )

            feedback = {
                "role": "user",
                "content": (
                    "RESULTADO DA EXECUÇÃO:\n"
                    + json.dumps(
                        result,
                        ensure_ascii=False,
                        indent=2
                    )
                    + "\n\n"
                    "Analise o resultado e decida o "
                    "próximo passo. Continue respeitando "
                    "todas as regras."
                ),
            }

            messages.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

            messages.append(
                feedback
            )

        # ----------------------------------------------------
        # CONCLUÍDO
        # ----------------------------------------------------

        elif action_type == "done":

            print()
            print("=" * 70)
            print("TAREFA CONCLUÍDA")
            print("=" * 70)

            print(
                json.dumps(
                    action,
                    ensure_ascii=False,
                    indent=2
                )
            )

            print()
            print(
                f"Chamadas API usadas: "
                f"{api_calls}/{MAX_API_CALLS}"
            )

            return

        # ----------------------------------------------------
        # PERGUNTA
        # ----------------------------------------------------

        elif action_type == "question":

            print()
            print(
                "Nemotron precisa de uma resposta:"
            )

            print(
                action.get(
                    "question",
                    ""
                )
            )

            answer = input("> ")

            messages.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

            messages.append(
                {
                    "role": "user",
                    "content":
                        "Resposta do usuário: "
                        + answer,
                }
            )

        # ----------------------------------------------------
        # TIPO DESCONHECIDO
        # ----------------------------------------------------

        else:

            print()
            print(
                "ERRO: tipo de ação desconhecido:",
                action_type
            )

            print(
                "Encerrando por segurança."
            )

            return

    print()
    print("=" * 70)
    print("LIMITE DE TURNOS ATINGIDO")
    print("=" * 70)

    print(
        f"Chamadas API usadas: "
        f"{api_calls}/{MAX_API_CALLS}"
    )


if __name__ == "__main__":
    main()
