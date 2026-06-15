# CONTRIBUTION GUIDELINE
## Welcome
Thank you! for considering contributing to the project whether you're here for fixing a bug, adding a new feature or improving our documentation.
your contribution is highly appreciated.

## Instruction
Follow the following steps on how you can contribute to the project.
1. **Fork the Repository:** Sart by forking the repository to your github.
2. **Clone the Repository:** Clone the forked repository from your github using `git clone https://github.com/{your-username}/Axiom-ML`.
3. **Setup dev environment:** use `python3 -m venv .venv` to make a virtual environment and activate it using `source .venv/bin/activate` and `pip install -e .[dev]` to install neccessary development packages
4. **Create a Branch:** Create a new branch for your feature or bug fix using `git checkout -b feature/your-feature-name`.
5. **Make changes:** Make sure the changes come with proper pre-writen documentation of the new feature in [README.md](./README.md) or github templates. And, it is recommended to write tests using pytest.
6. **Commit changes:** Commit your changes to the forked repository and concise commit message. refer to [commit guidelines](#commit-guidelines).
7. **Push to GitHub:** Push your commit changes to the forked repository using `git push origin feature/your-feature-name`.
8. **Submit a Pull Request:** Open a pull request from your forked repository to the main repository.

## Tests
Before submitting pull request, run atleast a single unit test either in jupyter notebook inside `/notebooks` or pytest file in `/tests`.
Make sure you've installed development dependencies
to run your pytest code, use the command.
`pytest`
in the root directory of the project.

## Commit guidelines
Commiting guidelines to be followed for proper and concise commits.
* `feat : Added Mean Absolute Error`
* `fix : Resolved the shape mismatch during backward pass (#001)`
* `doc : Updated the installation instructions`
* `test: Add unit test for the BCE loss function`

Examples of bad commit
* `Added new stuff`
* `Fixed a bug`
* `Updated code`
* `MAE updated`

## Getting Help
If you need help or have any question, feel free to contact me on:

[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/iprabhsidhu_)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:prabhdeepsidhu310@gmail.com)
