import os
import shutil
import pyfiglet


def clear_screen() -> None:
    os.system('cls' if os.name == 'nt' else 'clear')


def print_message(message: str) -> None:
    try:
        terminal_width = shutil.get_terminal_size().columns
    except (OSError, ValueError):
        terminal_width = 50
    print('\n' + '-' * terminal_width)
    print(f'\n{message}\n')
    print('-' * terminal_width + '\n')


def print_title(title: str) -> None:
    print(pyfiglet.figlet_format(title, font='3d-ascii'))


def print_done() -> None:
    print(pyfiglet.figlet_format('== Done ==', font='js_stick_letters'))
