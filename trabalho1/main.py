import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


CWD = Path(__name__).parent
GASTO_MENSAL_PATH = CWD / 'gasto-mensal.xls'
SALARIO_MINIMO_MENSAL_PATH = CWD / 'salario-minimo-mensal.xlsx'


# TODO: dowloand dataset automaticly with requests or selenium

MONTH_TO_NUMBER_MAP = {
        'Janeiro': 1,
        'Fevereiro': 2,
        'Março': 3,
        'Abril': 4,
        'Maio': 5,
        'Junho': 6,
        'Julho': 7,
        'Agosto': 8,
        'Setembro': 9,
        'Outubro': 10,
        'Novembro': 11,
        'Dezembro': 12
        }


class HandleDieeseDataset:
    """https://www.dieese.org.br"""

    CITIES_COLUMNS = [
        'Brasília', 'Campo Grande', 'Cuiabá', 'Goiânia',
        'Belo Horizonte', 'Rio de Janeiro', 'São Paulo', 'Vitória', 'Curitiba',
        'Florianópolis', 'Porto Alegre', 'Belém', 'Boa Vista', 'Macapá',
        'Manaus', 'Palmas', 'Porto Velho', 'Rio Branco', 'Aracaju',
        'Fortaleza', 'João Pessoa', 'Maceió', 'Natal', 'Recife', 'Salvador',
        'São Luís', 'Teresina', 'Macaé']
    MONTHLY_EXPENSE_DF_BASE_COLUMNS = ['Date (mm-yyyy)'] + CITIES_COLUMNS

    def __init__(self):
        self.monthly_expense_df = self.get_monthly_expense_df()
        self.basic_salary_df = self.get_basic_salary_df()
        self.merged_df = self.get_dfs_merged()

    def _get_df(self, file: Path, **kwargs):
        if not file.is_file():
            raise ValueError(f"Invalid file '{file}'")

        file_extension = file.suffix
        if file_extension == '.xlsx':
            df = pd.read_excel(file, engine='openpyxl', **kwargs)
        elif file_extension == '.xls':
            df = pd.read_excel(file, **kwargs)
        elif file_extension == '.csv':
            df = pd.read_csv(file, **kwargs)
        else:
            raise ValueError(f"Invalid file extension: '{file_extension}'")

        return df

    def get_monthly_expense_df(self):
        """Get monthly expense dataframe

        Note: Cesta básica Mensal

        Raw DF:
               Gasto Mensal - Total da Cesta Unnamed: 1    Unnamed: 2 Unnamed: 3  ... Unnamed: 25 Unnamed: 26 Unnamed: 27 Unnamed: 28
            0                            NaN   Brasília  Campo Grande     Cuiabá  ...    Salvador    São Luís    Teresina       Macaé
            1                        01-2000     106.66           NaN        NaN  ...       84.95         NaN         NaN         NaN
            ...
            276  (1) Série recalculada, conforme mudança metodo...        NaN           NaN        NaN  ...         NaN         NaN         NaN         NaN
            277  Tomada especial de preços a partir de abril de...        NaN           NaN        NaN  ...         NaN         NaN         NaN         NaN
        """
        df = self._get_df(GASTO_MENSAL_PATH, skiprows=1, skipfooter=3)
        df = df.rename(columns={'Unnamed: 0': 'Date (mm-yyyy)'})
        assert list(df.columns) == self.MONTHLY_EXPENSE_DF_BASE_COLUMNS

        return df

    def get_basic_salary_df(self):
        """Get basic salary dataframe

        Note: Salário mínimo mensal

        Raw DF:
                  Ano       Mes  Salário Mínimo  Salário Mínimo Reomendado
            0    2022  Setembro         1212.00                    6306.97
            1    2022    Agosto         1212.00                    6298.91
            2    2022     Julho         1212.00                    6388.55
            3    2022     Junho         1212.00                    6527.67
            4    2022      Maio         1212.00                    6535.40
            ..    ...       ...             ...                        ...
            333  1994  Novembro           70.00                     744.25
            334  1994   Outubro           70.00                     740.83
            335  1994  Setembro           70.00                     695.64
            336  1994    Agosto           64.79                     645.53
            337  1994     Julho           64.79                     590.33
            [338 rows x 4 columns]
        """
        df = self._get_df(SALARIO_MINIMO_MENSAL_PATH)
        return df

    def get_dfs_merged(self):
        df = self.monthly_expense_df.copy()
        for index, row in self.basic_salary_df.iterrows():
            month = MONTH_TO_NUMBER_MAP[row['Mes']]
            year = row['Ano']
            prefix = '0' if month < 10 else ''
            key_by_date = f'{prefix}{month}-{year}'
            df.loc[df['Date (mm-yyyy)'] == key_by_date, 'Salário Mínimo'] = (
                row['Salário Mínimo'])
            df.loc[df['Date (mm-yyyy)'] == key_by_date,
                   'Salário Mínimo Reomendado'] = (
                row['Salário Mínimo Reomendado'])

        assert not df['Salário Mínimo'].isnull().values.any(), (
                "Has empty values in 'Salário Mínimo' column")
        assert not df['Salário Mínimo Reomendado'].isnull().values.any(), (
                "Has empty values in 'Salário Mínimo Reomendado' column")

        return df

    def truncate(self, number, digits) -> float:
        nb_decimals = len(str(number).split('.')[1])
        if nb_decimals <= digits:
            return number
        stepper = 10.0 ** digits
        return math.trunc(stepper * number) / stepper

    def fill_average_expenses(self, decimal_places=5):
        def fill_available_fields(row):
            counter = 0
            for key, value in row.items():
                if key not in self.CITIES_COLUMNS:
                    continue

                if isinstance(value, float) and not np.isnan(value):
                    counter += 1

            return counter
            # return row.count() - 2  # not count date and total_fields columns

        def fill_average(row):
            # available_values = [
            #     value for value in row[:-2]  # exclude last two columns created
            #     if isinstance(value, float) and not np.isnan(value)]
            available_values = []
            for key, value in row.items():
                if key not in self.CITIES_COLUMNS:
                    continue

                if isinstance(value, float) and not np.isnan(value):
                    available_values.append(value)

            available_fields = row[-1]
            assert len(available_values) == available_fields

            return np.average(available_values)

        df = self.merged_df
        df['total_fields'] = df.apply(lambda row: len(row), axis=1)
        df['available_fields'] = df.apply(fill_available_fields, axis=1)
        df['average'] = df.apply(fill_average, axis=1)
        df['average_truncate'] = df.apply(
                lambda row: self.truncate(row.average, decimal_places), axis=1)
        df['average_round'] = df.apply(
                lambda row: round(row.average, decimal_places), axis=1)
        return df

    def real_error(self, p, p_):
        """Real Error

        Args:
            p
            p_: Approximation of `p`.

        Returns:
            A float number.
        """
        return p - p_

    def absolute_error(self, p, p_):
        """Real Error

        Args:
            p
            p_: Approximation of `p`.

        Returns:
            A float number.
        """
        return abs(self.real_error(p, p_))

    def relative_error(self, p, p_):
        """Real Error

        Args:
            p
            p_: Approximation of `p`.

        Returns:
            A float number.
        """
        return self.absolute_error(p, p_) / abs(p)

    def fill_errors_info(self):
        df = self.merged_df
        error_functions = [
                self.real_error, self.absolute_error, self.relative_error]
        suffixes = {'truncate': 'average_truncate', 'round': 'average_round'}
        for func in error_functions:
            for sufix, p_arg_name in suffixes.items():
                column_name = f"{func.__name__}_{sufix}"
                df[column_name] = df.apply(
                    lambda row: func(row.average, getattr(row, p_arg_name)),
                    axis=1)

        expected_colums = self.MONTHLY_EXPENSE_DF_BASE_COLUMNS + [
            'Salário Mínimo', 'Salário Mínimo Reomendado'] + [
            'total_fields', 'available_fields', 'average', 'average_truncate',
            'average_round', 'real_error_truncate', 'real_error_round',
            'absolute_error_truncate', 'absolute_error_round',
            'relative_error_truncate', 'relative_error_round']
        assert list(df.columns) == expected_colums
        return df

    def fill_percentage_between_basic_salary_and_monthly_expense(self):
        """Preenche relação entre salário mínimo e cesta básica

        Regra de 3:
            salario_minimo - 100
            media - x
            x = media * 100 / salario_minimo
        """
        def fill_relation(row):
            basic_salary = row['Salário Mínimo']
            average = row['average']
            percentage = (average * 100) / basic_salary
            return percentage

        df = self.merged_df
        df['percentage_between_basic_salary_and_monthly_expense'] = (
                df.apply(fill_relation, axis=1))
        return df

    def process(self):
        self.fill_average_expenses()
        self.fill_errors_info()
        self.fill_percentage_between_basic_salary_and_monthly_expense()

    def plot_polynomial_regression(self):
        df = self.merged_df
        dates = df['Date (mm-yyyy)'].values
        dates_parsed = [idx + 10 for idx, _ in enumerate(dates)]
        percentages = df[
            'percentage_between_basic_salary_and_monthly_expense'].values

        x = dates_parsed
        y = percentages

        model2 = np.poly1d(np.polyfit(x, y, 2))
        model3 = np.poly1d(np.polyfit(x, y, 3))
        model4 = np.poly1d(np.polyfit(x, y, 4))
        model5 = np.poly1d(np.polyfit(x, y, 5))

        y2 = model2(x)
        y3 = model3(x)
        y4 = model4(x)
        y5 = model5(x)

        # Plotting
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].scatter(x, y)
        ax[0, 0].plot(x, y2, color='red')
        ax[0, 0].set_title('Grau 2')

        ax[0, 1].scatter(x, y)
        ax[0, 1].plot(x, y3, color='red')
        ax[0, 1].set_title('Grau 3')

        ax[1, 0].scatter(x, y)
        ax[1, 0].plot(x, y4, color='red')
        ax[1, 0].set_title('Grau 4')

        ax[1, 1].scatter(x, y)
        ax[1, 1].plot(x, y5, color='red')
        ax[1, 1].set_title('Grau 5')
        plt.tight_layout()
        plt.show()

        # Printing error
        MAE2 = mean_absolute_error(y, y2)
        MAE3 = mean_absolute_error(y, y3)
        MAE4 = mean_absolute_error(y, y4)
        MAE5 = mean_absolute_error(y, y5)

        print("MAE (grau=2) = {:0.4f}".format(MAE2))
        print("MAE (grau=3) = {:0.4f}".format(MAE3))
        print("MAE (grau=4) = {:0.4f}".format(MAE4))
        print("MAE (grau=5) = {:0.4f}".format(MAE5))

        RMSE2 = np.sqrt(mean_squared_error(y, y2))
        RMSE3 = np.sqrt(mean_squared_error(y, y3))
        RMSE4 = np.sqrt(mean_squared_error(y, y4))
        RMSE5 = np.sqrt(mean_squared_error(y, y5))

        print("RMSE (grau=2) = {:0.4f}".format(RMSE2))
        print("RMSE (grau=3) = {:0.4f}".format(RMSE3))
        print("RMSE (grau=4) = {:0.4f}".format(RMSE4))
        print("RMSE (grau=5) = {:0.4f}".format(RMSE5))


def main():
    instance = HandleDieeseDataset()
    instance.process()
    instance.plot_polynomial_regression()


if __name__ == '__main__':
    main()
