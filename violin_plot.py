"""
This code generates a violin plot using the seaborn library. The plot uses the Iris dataset for 
demonstration, and includes example annotations for significance based on statistical testing. 
The appearance of the plot is customized and the final figure is saved.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import starbars
import pypalettes

# suppress pandas warning
pd.options.mode.chained_assignment = None


def generate_plot(df,
                  group_variable,
                  dependent_variable,
                  group_variable_order,
                  group_variable_label=None,
                  dependent_variable_label=None,
                  dependent_variable_range=None,
                  palette_list=None,
                  super_title = None):
    """
    Generates a violin plot with statistical annotations.
    """

    sns.set(style="whitegrid", font_scale=1.6)

    # default color palette assignment if not provided
    if palette_list is None:
        palette_list = sns.color_palette("Blues", len(group_variable_order))

    with plt.rc_context({'axes.edgecolor': 'black'}):

        fig = plt.figure(figsize=(5, 7))

        # violin plot
        ax = sns.violinplot(data=df, x=group_variable, y=dependent_variable,
                            inner='quartile',
                            order=group_variable_order, linewidth=3,
                            hue=group_variable,
                            palette=['white'] * len(group_variable_order),
                            legend=False)
        # recolor violin edges
        for t, _ in enumerate(palette_list):
            ax.collections[t].set_edgecolor(palette_list[t])
        # recolor violin quartile lines
        line_count = 0
        for line_var in ax.lines:
            if line_count % 3 == 1:
                line_var.set_linestyle('--')
                line_var.set_linewidth(2)
            else:
                line_var.set_linestyle(':')
                line_var.set_linewidth(2)
            line_var.set_color('black')
            line_var.set_alpha(1)
            line_count += 1

        # # uncomment to add swarm plot to violins
        # # swarm plot
        # ax = sns.swarmplot(data=df, x=group_variable, y=dependent_variable,
        #                    order=group_variable_order,
        #                    size=4,
        #                    facecolors='white',
        #                    edgecolor='k',
        #                    linewidth=2,
        #                    zorder=2)

        ax.axhline(y=0, color='k', linestyle=':', linewidth=2)

        # annotation significance method 1 (using mannwhitneyu test on the data)
        current_df = df
        filtered_df1 = current_df[current_df[group_variable] == group_variable_order[0]]
        filtered_df2 = current_df[current_df[group_variable] == group_variable_order[1]]
        data1 = filtered_df1[dependent_variable].to_numpy()
        data2 = filtered_df2[dependent_variable].to_numpy()
        _, p_value = mannwhitneyu(data1, data2)

        # assign annotation values
        annotations = [(group_variable_order[0], group_variable_order[1], p_value)]

        if dependent_variable_range is not None:
            # annotation positioning adjustment
            specified_lower_y_limit = dependent_variable_range[0]
            specified_upper_y_limit = dependent_variable_range[1]
            current_lower_y_limit = ax.get_ylim()[0]
            current_upper_y_limit = ax.get_ylim()[1]
            y_range_ratio = \
                (specified_upper_y_limit - specified_lower_y_limit) / \
                    (current_upper_y_limit - current_lower_y_limit)
        else:
            y_range_ratio = 1
        # annotation plotting
        starbars.draw_annotation(annotations, fontsize=20,
                                    bar_gap=0.03 * y_range_ratio,
                                    text_distance=0.08 * y_range_ratio)

        # set y limits
        if dependent_variable_range is not None:
            # set y ranges
            ax.set(ylim=(dependent_variable_range[0], dependent_variable_range[1]))

        # set x and y labels
        ax.set_xlabel('')
        if group_variable_label is not None:
            ax.set_xticks(range(len(group_variable_label)), labels=group_variable_label)
        if dependent_variable_label is not None:
            ax.set_ylabel(dependent_variable_label)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='both', colors='black')

        # hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # only show ticks on the left and bottom spines
        ax.get_yaxis().tick_left()
        ax.get_xaxis().tick_bottom()
        # puts ticks inside the axes
        ax.tick_params(direction='in')

        # set the color of the axis labels
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')

        # remove grid lines
        ax.grid(False)

        # adjust subplots spacing
        # if subplots are added, can include, for e.g., 'wspace=0.4, hspace=0.4'
        # to control padding between subplots
        plt.subplots_adjust(bottom=0.3, top=0.85, left=0.29, right=0.81)

        # add global title
        if super_title is not None:
            fig.suptitle(super_title, fontsize="large", color="k")

        # draw line and text below group labels
        trans = ax.get_xaxis_transform()
        ax.plot([0.8, 2.3], [-.36, -.36], color="k", transform=trans, clip_on=False, linewidth=2)
        plt.figtext(0.645, 0.085, "non-setosa", ha="center", va="top", fontsize=20, color="k")


if __name__ == '__main__':

    # --- read data ---
    EXAMPLE_DATA_PATH = r'.\Iris.csv'
    example_data_df = pd.read_csv(EXAMPLE_DATA_PATH)

    # --- variables setup ---
    # assign data-specific variables
    example_group_variable = example_data_df.columns.tolist()[-1]
    example_dependent_variable = example_data_df.columns.tolist()[2]
    example_group_variable_order = example_data_df[example_group_variable].unique().tolist()
    # palette setup
    cmap = pypalettes.load_cmap("Halichoeres_brasiliensis",
                                keep_first_n=len(example_group_variable_order))
    pypalettes_list = cmap.colors # return colors as a list of hexadecimal values

    # --- plot data ---
    generate_plot(example_data_df,
                  example_group_variable,
                  example_dependent_variable,
                  example_group_variable_order,
                  group_variable_label=['Iris setosa', 'Iris versicolor', 'Iris virginica'],
                  dependent_variable_label=r'Sepal Width $\mathregular{[cm]}$',
                  dependent_variable_range=[-2, 8],
                  palette_list=pypalettes_list)

    # save figure
    FILE_DESTINATION = r'.\figure'
    plt.savefig(os.path.join(FILE_DESTINATION + '.pdf').replace("\\", "/"), format="pdf")
    plt.savefig(os.path.join(FILE_DESTINATION + '.png').replace("\\", "/"), dpi=300)
    plt.close()
