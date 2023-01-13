"""
Licensed under the GNU General Public License, Version 3.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.gnu.org/licenses/gpl-3.0.html.en

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de
"""
from arch_opt_exp.problems.so_mo import MIMOGoldstein
from arch_opt_exp.algorithms.sbo import get_sbo_krg
from pymoo.optimize import minimize


def test_sbo():
    algorithm = get_sbo_krg(init_size=20)
    problem = MIMOGoldstein()
    result = minimize(problem, algorithm, ('n_eval', 30))
    assert len(result.opt) > 0
